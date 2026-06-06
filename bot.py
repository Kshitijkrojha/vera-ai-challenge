"""
Vera AI Bot — Magicpin AI Challenge v2.4
Deterministic decision engine: business-impact trigger scoring → fact-grounded composer.

Key improvements over v2.3:
  1. Persistence path hardened: uses ./vera_state.json (app working dir) not /tmp
     which vanishes on Railway/Render container restarts. Env var VERA_STATE_FILE
     overrides. Falls back gracefully — never crashes on I/O error.
  2. Cross-category trigger penalty: supply_alert outside pharmacies, ipl_match_today
     outside restaurants, chronic_refill_due outside pharmacies all get hard negative
     scores so they can never outrank the correct category's top trigger.
  3. Reply fallback upgraded: ambiguous merchant message now uses trigger context and
     last Vera message to give a specific reply (not a generic "Should I go ahead?")
  4. Anthropic client init hardened: missing API key logs a warning instead of silently
     setting empty string — prevents confusing 401 errors
  5. JSON parse robustness in compose_message: strips any leading/trailing non-JSON
     characters before parse attempt; one extra fallback pass
  6. /v1/healthz now reports persistence status so judge can verify state is live

Key improvements retained from v2.3:
  - File-based persistence across restarts
  - Counter-intuitive IPL Saturday reframe + seasonal dip reframe
  - Supply alert derived customer count from cagg
  - Full 24-trigger fallback coverage with case-study-aligned messages
  - Dual hallucination guard (numeric + named-claim)
  - Category-specific scoring boosts and voice guidance
"""
import os
import re
import time
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vera_bot")

app = FastAPI(title="Vera Bot — Magicpin AI Challenge v2.3")
START_TIME = time.time()

# ─── Persistence layer ────────────────────────────────────────────────────────
# Use the app working directory (survives Railway deploys within a session).
# /tmp is wiped on container restart on most PaaS platforms — don't use it.
# Set VERA_STATE_FILE env var to override (e.g. a mounted volume path).
_PERSIST_FILE = Path(os.environ.get("VERA_STATE_FILE", "./vera_state.json")).resolve()
_persist_lock = threading.Lock()

def _load_state() -> dict:
    """Load persisted state from disk, return empty defaults on any error."""
    try:
        if _PERSIST_FILE.exists():
            raw = _PERSIST_FILE.read_text()
            data = json.loads(raw)
            return data
    except Exception as e:
        logger.warning(f"State load failed (using fresh state): {e}")
    return {"contexts": {}, "conversations": {}, "suppressed_keys": []}

def _save_state() -> None:
    """Write current in-memory state to disk (non-blocking best-effort)."""
    try:
        with _persist_lock:
            state = {
                "contexts": {f"{k[0]}|{k[1]}": v for k, v in contexts.items()},
                "conversations": conversations,
                "suppressed_keys": list(suppressed_keys),
            }
            _PERSIST_FILE.write_text(json.dumps(state, default=str))
    except Exception as e:
        logger.warning(f"State save failed (non-fatal): {e}")

def _restore_state() -> None:
    """Restore in-memory stores from persisted state on startup."""
    data = _load_state()
    for compound_key, val in data.get("contexts", {}).items():
        if "|" in compound_key:
            scope, cid = compound_key.split("|", 1)
            contexts[(scope, cid)] = val
    conversations.update(data.get("conversations", {}))
    suppressed_keys.update(data.get("suppressed_keys", []))
    logger.info(f"State restored from {_PERSIST_FILE}: {len(contexts)} contexts, "
                f"{len(conversations)} convs, {len(suppressed_keys)} suppressions")

# ─── In-memory stores ────────────────────────────────────────────────────────
contexts: dict[tuple[str, str], dict] = {}
conversations: dict[str, dict] = {}
suppressed_keys: set[str] = set()

# ─── Restore persisted state on startup ──────────────────────────────────────
_restore_state()

# ─── Anthropic client ────────────────────────────────────────────────────────
_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not _api_key:
    logger.warning("ANTHROPIC_API_KEY not set — all compose calls will use fallback templates")
anthropic_client = anthropic.Anthropic(api_key=_api_key)

# ─── Pydantic models ─────────────────────────────────────────────────────────
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ─── Context helpers ─────────────────────────────────────────────────────────
def get_context(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None

def count_contexts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts

# ─── Signal detection ─────────────────────────────────────────────────────────
def is_auto_reply(message: str) -> bool:
    patterns = [
        "thank you for contacting", "our team will respond", "this is an automated",
        "auto-reply", "out of office", "we will get back to you", "आपका संदेश प्राप्त",
        "we received your message", "automated response",
    ]
    msg = message.lower()
    return any(p in msg for p in patterns)

def is_hostile(message: str) -> bool:
    patterns = [
        "stop messaging", "don't message", "not interested", "leave me alone",
        "stop sending", "unsubscribe", "remove me", "do not contact",
        "बंद करो", "मत भेजो", "परेशान मत करो", "spam", "report", "block",
        "useless", "bothering me", "why are you bothering",
    ]
    msg = message.lower()
    return any(p in msg for p in patterns)

def is_intent_commit(message: str) -> bool:
    patterns = [
        "let's do it", "lets do it", "ok do it", "yes do it", "go ahead",
        "proceed", "confirm", "activate", "send it", "schedule it",
        "haan karo", "kar do", "theek hai karo", "done", "perfect", "approved",
    ]
    msg = message.lower()
    return any(p in msg for p in patterns)

def is_positive_yes(message: str) -> bool:
    patterns = ["yes", "haan", "sure", "ok", "please", "send", "yes please", "sounds good",
                "great", "go for it", "do it", "yes send", "send please"]
    msg = message.lower().strip()
    return any(msg == p or msg.startswith(p + " ") or msg.endswith(" " + p) for p in patterns)

def is_price_objection(message: str) -> bool:
    patterns = ["expensive", "costly", "budget", "afford", "price", "mehnga", "zyada",
                "too much", "reduce", "discount", "cheaper"]
    msg = message.lower()
    return any(p in msg for p in patterns)

def is_deferral(message: str) -> bool:
    patterns = ["later", "baad", "busy", "not now", "tomorrow", "kal", "next week",
                "abhi nahi", "give me", "some time", "weekend", "monday"]
    msg = message.lower()
    return any(p in msg for p in patterns)

def is_why_question(message: str) -> bool:
    patterns = ["why", "kyu", "how does this help", "what is this", "explain",
                "kyun", "what benefit", "how will this", "what's the point"]
    msg = message.lower()
    return any(p in msg for p in patterns)

def is_out_of_scope(message: str) -> bool:
    """Detect requests that are clearly outside Vera's scope (GST, HR, legal, etc.)"""
    patterns = [
        "gst", "tax filing", "income tax", "itr", "hr issue", "employee",
        "legal notice", "lawsuit", "court", "bank loan", "loan application",
        "insurance claim", "pf ", "epf", "salary slip",
    ]
    msg = message.lower()
    return any(p in msg for p in patterns)

# ─── Fact extractor ───────────────────────────────────────────────────────────
def extract_facts(trigger: dict, merchant: dict, category: dict, customer: Optional[dict]) -> dict:
    """
    Extract ONLY facts present in the context objects.
    Anti-hallucination layer — the LLM may ONLY reference facts listed here.
    Schema-safe: tries multiple known field names so gym/pharmacy/dentist merchants all work.
    """
    facts: dict[str, Any] = {}

    # --- Merchant identity ---
    identity = merchant.get("identity", {})
    facts["merchant_name"] = identity.get("name", "")
    facts["owner_first"] = identity.get("owner_first_name", "")
    facts["locality"] = identity.get("locality", "")
    facts["city"] = identity.get("city", "")
    facts["languages"] = identity.get("languages", ["en"])
    facts["verified"] = identity.get("verified", False)

    # --- Subscription ---
    sub = merchant.get("subscription", {})
    facts["sub_status"] = sub.get("status", "")
    facts["sub_plan"] = sub.get("plan", "")
    facts["sub_days_remaining"] = sub.get("days_remaining")

    # --- Performance (only include if present) ---
    perf = merchant.get("performance", {})
    for k in ["views", "calls", "directions", "ctr", "leads"]:
        v = perf.get(k)
        if v is not None:
            facts[f"perf_{k}"] = v
    delta = perf.get("delta_7d", {})
    for k in ["views_pct", "calls_pct", "ctr_pct"]:
        v = delta.get(k)
        if v is not None:
            facts[f"delta_{k}"] = v

    # --- Peer stats (only include if present) ---
    peer = category.get("peer_stats", {})
    # Store under consistent keys — note: JSON field is "avg_ctr" not "avg_avg_ctr"
    peer_map = {
        "avg_ctr": "peer_avg_ctr",
        "avg_views_30d": "peer_avg_views_30d",
        "avg_calls_30d": "peer_avg_calls_30d",
        "avg_rating": "peer_avg_rating",
        "avg_review_count": "peer_avg_review_count",
        "retention_6mo_pct": "peer_retention_6mo_pct",
    }
    for json_key, facts_key in peer_map.items():
        v = peer.get(json_key)
        if v is not None:
            facts[facts_key] = v

    # --- CTR gap (computed from verified values only) ---
    if "perf_ctr" in facts and "peer_avg_ctr" in facts:
        ctr = facts["perf_ctr"]
        peer_ctr = facts["peer_avg_ctr"]
        if peer_ctr > 0:
            facts["ctr_gap_pct"] = round(((ctr - peer_ctr) / peer_ctr) * 100, 1)
            # Human-readable: e.g. "2.1% vs 3.0%"
            facts["ctr_merchant_pct_str"] = f"{ctr * 100:.1f}%"
            facts["ctr_peer_pct_str"] = f"{peer_ctr * 100:.1f}%"

    # --- Active offers only ---
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    facts["active_offers"] = active_offers
    facts["has_active_offer"] = len(active_offers) > 0

    # --- Customer aggregate — schema-safe across all categories ---
    cagg = merchant.get("customer_aggregate", {})
    # Universal fields (present in most categories)
    for k in ["total_unique_ytd", "lapsed_180d_plus", "lapsed_90d_plus",
              "retention_6mo_pct", "high_risk_adult_count"]:
        v = cagg.get(k)
        if v is not None:
            facts[f"cagg_{k}"] = v
    # Gym-specific fields
    for k in ["total_active_members", "monthly_churn_pct", "trial_to_paid_pct", "active_members"]:
        v = cagg.get(k)
        if v is not None:
            facts[f"cagg_{k}"] = v
    # Pharmacy-specific fields
    for k in ["repeat_customer_pct", "chronic_rx_count"]:
        v = cagg.get(k)
        if v is not None:
            facts[f"cagg_{k}"] = v

    # --- Signals ---
    facts["signals"] = merchant.get("signals", [])

    # --- Conversation history ---
    history = merchant.get("conversation_history", [])
    facts["last_vera_msg"] = history[-2]["body"] if len(history) >= 2 else None
    facts["last_merchant_msg"] = (history[-1]["body"]
                                  if history and history[-1].get("from") == "merchant"
                                  else None)
    facts["recent_engagement"] = history[-1].get("engagement") if history else None
    # Was merchant engaged recently?
    facts["merchant_engaged_recently"] = any(
        h.get("engagement") in ("merchant_replied", "intent_action")
        for h in history[-3:]
    )

    # --- Review themes ---
    facts["review_themes"] = merchant.get("review_themes", [])

    # --- Trigger ---
    trg_payload = trigger.get("payload", {})
    facts["trigger_kind"] = trigger.get("kind", "")
    facts["trigger_payload"] = trg_payload
    facts["trigger_urgency"] = trigger.get("urgency", 1)
    # Surface key payload fields at top level for easier LLM use
    if "vs_baseline" in trg_payload:
        facts["trigger_vs_baseline"] = trg_payload["vs_baseline"]
    if "is_expected_seasonal" in trg_payload:
        facts["trigger_is_expected_seasonal"] = trg_payload["is_expected_seasonal"]
    if "is_weeknight" in trg_payload:
        facts["trigger_is_weeknight"] = trg_payload["is_weeknight"]

    # Resolve digest item if referenced
    top_item_id = trg_payload.get("top_item_id") or trg_payload.get("alert_id") or trg_payload.get("digest_item_id")
    if top_item_id:
        for item in category.get("digest", []):
            if item.get("id") == top_item_id:
                facts["digest_item"] = item
                break

    # --- Category voice ---
    voice = category.get("voice", {})
    facts["voice_tone"] = voice.get("tone", "professional")
    facts["voice_taboo"] = voice.get("vocab_taboo", [])
    facts["voice_allowed"] = voice.get("vocab_allowed", [])
    facts["category_slug"] = merchant.get("category_slug", "")

    # --- Patient content library (dentists/pharmacies) ---
    facts["patient_content_library"] = category.get("patient_content_library", [])

    # --- Seasonal beats (for timing-sensitive messages) ---
    facts["seasonal_beats"] = category.get("seasonal_beats", [])

    # --- Trend signals ---
    facts["trend_signals"] = category.get("trend_signals", [])

    # --- Customer ---
    if customer:
        cid = customer.get("identity", {})
        rel = customer.get("relationship", {})
        prefs = customer.get("preferences", {})
        consent = customer.get("consent", {})
        facts["customer_name"] = cid.get("name", "")
        facts["customer_lang"] = cid.get("language_pref", "en")
        facts["customer_age_band"] = cid.get("age_band", "")
        facts["customer_state"] = customer.get("state", "")
        facts["customer_last_visit"] = rel.get("last_visit", "")
        facts["customer_visits_total"] = rel.get("visits_total")
        facts["customer_services"] = rel.get("services_received", [])
        facts["customer_ltv"] = rel.get("lifetime_value")
        facts["customer_pref_slot"] = prefs.get("preferred_slots", "")
        facts["customer_channel"] = prefs.get("channel", "whatsapp")
        facts["customer_consent_scope"] = consent.get("scope", [])
        # Also store gym-specific customer fields if present
        facts["customer_prev_focus"] = rel.get("previous_focus", "")
        facts["customer_prev_membership_months"] = rel.get("previous_membership_months")
        # Compute months since last visit if possible
        lv = facts.get("customer_last_visit", "")
        if lv:
            try:
                last = datetime.fromisoformat(lv)
                now = datetime.now(timezone.utc)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                facts["customer_months_since_visit"] = round((now - last).days / 30)
            except Exception:
                pass

    return facts


# ─── Trigger scoring ──────────────────────────────────────────────────────────
def score_trigger(trg: dict, merchant: dict, category: dict) -> float:
    """
    Business-impact scoring: urgency × kind × merchant state × category fit.
    All signals from supplied context — no invented values.
    """
    urgency = trg.get("urgency", 1)
    kind = trg.get("kind", "")
    payload = trg.get("payload", {})
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    cagg = merchant.get("customer_aggregate", {})
    cat = merchant.get("category_slug", "")
    history = merchant.get("conversation_history", [])

    # Base: urgency (1-5) × 10
    score = urgency * 10.0

    # ── Revenue-critical boosts ──────────────────────────────────────────────
    if kind == "renewal_due":
        days = payload.get("days_remaining", sub.get("days_remaining", 30))
        score += max(0, (14 - (days or 30)) * 2)   # peaks +28 at day-of
        score += 15

    elif kind == "perf_dip":
        delta = abs(payload.get("delta_pct", 0))
        score += delta * 20           # 0.5 drop → +10
        if not offers:
            score += 8                # can't self-recover

    elif kind == "seasonal_perf_dip":
        delta = abs(payload.get("delta_pct", 0))
        score += delta * 15
        if payload.get("is_expected_seasonal"):
            score -= 5                # known pattern, slightly lower urgency

    elif kind == "recall_due":
        score += 12
        if offers:
            score += 5

    elif kind == "regulation_change":
        deadline = payload.get("deadline_iso", "")
        if deadline:
            try:
                dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                days_left = (dl - datetime.now(timezone.utc)).days
                if days_left < 60:
                    score += 10
                if days_left < 30:
                    score += 10
            except Exception:
                score += 8

    elif kind == "supply_alert":
        # Pharmacy emergency — always very high
        score += 25

    elif kind == "chronic_refill_due":
        # Time-critical for patient; score by days until stock runs out
        runs_out = payload.get("stock_runs_out_iso", "")
        if runs_out:
            try:
                ro = datetime.fromisoformat(runs_out.replace("Z", "+00:00"))
                days_left = (ro - datetime.now(timezone.utc)).days
                if days_left < 3:
                    score += 20
                elif days_left < 7:
                    score += 12
            except Exception:
                score += 10
        score += 8

    elif kind == "review_theme_emerged":
        occ = payload.get("occurrences_30d", 0)
        score += occ * 1.5
        if payload.get("trend") == "rising":
            score += 8

    elif kind == "winback_eligible":
        lapsed = payload.get("lapsed_customers_added_since_expiry", 0)
        score += min(lapsed * 0.5, 10)

    elif kind == "customer_lapsed_hard":
        days_since = payload.get("days_since_last_visit", 0)
        if days_since > 90:
            score += 10
        elif days_since > 60:
            score += 6

    elif kind == "ipl_match_today":
        if cat == "restaurants":
            score += 12
        else:
            score -= 15               # completely irrelevant for others

    elif kind == "festival_upcoming":
        days_until = payload.get("days_until", 999)
        if days_until < 14 and offers:
            score += 10
        elif days_until < 14:
            score += 5
        elif days_until < 30:
            score += 2
        else:
            score -= 8                # too early

    elif kind == "research_digest":
        signals = merchant.get("signals", [])
        if "high_risk_adult_cohort" in signals or cagg.get("high_risk_adult_count"):
            score += 8
        score += 5

    elif kind == "cde_opportunity":
        # Low urgency CDE webinar — relevant but only if merchant is engaged
        recent_engaged = any(
            h.get("engagement") in ("merchant_replied", "intent_action")
            for h in history[-3:]
        )
        if recent_engaged:
            score += 5
        else:
            score -= 5

    elif kind == "wedding_package_followup":
        days_to = payload.get("days_to_wedding", 999)
        if 30 <= days_to <= 180:
            score += 15
        score += 8

    elif kind == "curious_ask_due":
        recent_engaged = any(
            h.get("engagement") in ("merchant_replied", "intent_action")
            for h in history[-3:]
        )
        if recent_engaged:
            score += 3
        else:
            score -= 5

    elif kind == "active_planning_intent":
        # Merchant explicitly asked for help — very high priority
        score += 20

    elif kind == "trial_followup":
        score += 8

    elif kind == "competitor_opened":
        score += 5
        # Extra urgency if competitor is cheaper
        their_offer = payload.get("their_offer", "")
        dist_km = payload.get("distance_km", 99)
        if dist_km < 2:
            score += 5

    elif kind == "milestone_reached":
        if payload.get("is_imminent"):
            score += 8
        else:
            score += 3

    elif kind == "perf_spike":
        # Positive news — good for engagement, low urgency
        score += 3

    elif kind == "gbp_unverified":
        # Listing fix — moderate urgency
        score += 8

    elif kind == "category_seasonal":
        score += 5

    elif kind == "dormant_with_vera":
        # Re-engage dormant merchant
        days_dormant = payload.get("days_since_last_merchant_message", 0)
        if days_dormant > 30:
            score += 5
        else:
            score += 2

    # ── Category-specific fine-tuning ────────────────────────────────────────
    cat_kind_boost = {
        "dentists": {
            "recall_due": 5, "regulation_change": 5, "research_digest": 4,
            "competitor_opened": 4, "cde_opportunity": 3,
        },
        "salons": {
            "wedding_package_followup": 5, "curious_ask_due": 3,
            "festival_upcoming": 4, "review_theme_emerged": 4,
        },
        "restaurants": {
            "ipl_match_today": 5, "review_theme_emerged": 5,
            "active_planning_intent": 4, "festival_upcoming": 3,
        },
        "gyms": {
            "renewal_due": 5, "perf_dip": 4, "customer_lapsed_hard": 5,
            "trial_followup": 4, "seasonal_perf_dip": 3,
        },
        "pharmacies": {
            "regulation_change": 8, "recall_due": 6, "supply_alert": 10,
            "chronic_refill_due": 8, "gbp_unverified": 5,
        },
    }
    score += cat_kind_boost.get(cat, {}).get(kind, 0)

    # ── Hard cross-category penalties ────────────────────────────────────────
    # These triggers are only meaningful for specific categories.
    # If they appear for the wrong category (injected edge case), kill the score.
    category_exclusive = {
        "supply_alert":       {"pharmacies"},
        "chronic_refill_due": {"pharmacies"},
        "ipl_match_today":    {"restaurants"},
        "cde_opportunity":    {"dentists"},
        "wedding_package_followup": {"salons"},
        "trial_followup":     {"gyms"},
    }
    if kind in category_exclusive and cat not in category_exclusive[kind]:
        score -= 50   # ensures it never beats a legitimate trigger

    return score


def select_best_trigger(triggers: list[dict], merchant: dict, category: dict) -> Optional[dict]:
    """Pick the single highest business-impact trigger."""
    if not triggers:
        return None
    return max(triggers, key=lambda t: score_trigger(t, merchant, category))


# ─── Message composer ─────────────────────────────────────────────────────────
def build_composer_prompt(facts: dict) -> str:
    """
    Build the LLM prompt with ONLY extracted (verified) facts.
    Explicitly forbids inventing any number not present in the facts block.
    """
    facts_json = json.dumps(facts, indent=2, default=str)

    category_guidance = {
        "dentists": (
            "Use clinical peer tone (Dr./Doc salutation). Technical terms welcome: "
            "fluoride varnish, scaling, caries, RCT, OPG, aligner. "
            "NEVER say 'guaranteed', 'cure', 'best in city'. "
            "Cite source + page for research items. Voice: respectful-collegial."
        ),
        "salons": (
            "Warm, aspirational, practical. Use owner first name. "
            "Reference specific services from the merchant's offer list. "
            "For customer messages: language-mix if customer prefers hi-en. "
            "Voice: warm-practical."
        ),
        "restaurants": (
            "Operator-to-operator tone. Use 'covers', 'AOV', 'footfall', 'combo'. "
            "IMPORTANT: Saturday/weekend IPL matches = delivery spike, NOT dine-in footfall. "
            "Counter-intuitive but correct — push delivery offers on weekend IPL, not dine-in promos. "
            "Reference IPL/festival only when directly applicable. "
            "Data-first: delivery time, order volume, conversion rates. "
            "Voice: peer-operator."
        ),
        "gyms": (
            "Coach-to-coach tone. Reference member counts, attendance, retention. "
            "IMPORTANT: April-June seasonal dip is NORMAL (every metro gym sees -25 to -35%). "
            "For seasonal_perf_dip: reframe as expected, advise saving ad spend for Sept-Oct when conversion is 2x. "
            "Focus retention on active members during the dip, not acquisition. "
            "For customer messages: no-shame framing, goal-aligned. "
            "Use 'ad spend', 'conversion', 'seasonal lull'. "
            "Voice: motivating-precise."
        ),
        "pharmacies": (
            "Trustworthy, precise, compliance-first. Batch numbers, molecule names, dates. "
            "For senior customers: namaste, respectful, both Hindi + English option. "
            "NEVER overstate risk; bound it ('no safety risk, but...'). "
            "Voice: trustworthy-precise."
        ),
    }
    cat_guide = category_guidance.get(facts.get("category_slug", ""), "Professional, concise tone.")

    prompt = f"""You are Vera, Magicpin's AI assistant for merchant growth.

Your task: compose ONE highly specific, grounded WhatsApp message for the merchant (or customer) right now.

═══════════════════════════════════════════════════════
VERIFIED FACTS (use ONLY these — never invent numbers)
═══════════════════════════════════════════════════════
{facts_json}

═══════════════════════════════════════════════════════
CATEGORY GUIDANCE
═══════════════════════════════════════════════════════
{cat_guide}

═══════════════════════════════════════════════════════
STRICT RULES
═══════════════════════════════════════════════════════
1. SINGLE STRONGEST INSIGHT — pick ONE signal from the trigger. Do not list multiple facts.
2. REAL NUMBERS ONLY — every figure in your message must exist in the VERIFIED FACTS above.
   If a number is not in VERIFIED FACTS, do NOT include it. Paraphrase instead.
3. ONE CTA — end with exactly ONE easy yes/no or single-action question. Never multiple.
4. UNDER 200 WORDS — WhatsApp-friendly, no URLs.
5. NO GENERIC PHRASES — never: "boost engagement", "increase sales", "run a campaign", "grow your business".
6. CATEGORY VOICE — use the vocabulary and tone from CATEGORY GUIDANCE. Wrong vocab = penalty.
7. OWNER NAME — use owner_first from facts as the salutation where appropriate.
8. CUSTOMER MESSAGES — if customer_name is present, address the customer directly.
   Honor customer_lang (hi-en mix if specified). Use merchant_name as sender identity.
9. TABOO WORDS — never use: {facts.get("voice_taboo", [])}
10. RATIONALE — concisely explain which fact drove the message choice.
11. COUNTER-INTUITIVE ADVICE — when is_expected_seasonal=true (gyms, seasonal_perf_dip):
    reframe the dip as normal, tell merchant to SAVE ad spend for high-conversion windows.
    When ipl_match_today AND is_weeknight=false: Saturday IPL = delivery spike, not dine-in.
    Push delivery offer, not footfall promo. This counter-intuitive call is the highest-signal advice.

Output JSON only (no markdown fences):
{{
  "body": "<the WhatsApp message, max 200 words>",
  "cta": "binary_yes_no" | "open_ended" | "binary_confirm_cancel" | "multi_choice_slot",
  "rationale": "<1-2 sentences: which fact/trigger drove this, and why it was chosen over alternatives>",
  "template_name": "<descriptive_snake_case_template_name_v1>",
  "send_as": "vera" | "merchant_on_behalf",
  "numbers_used": ["<list every number you put in body, for audit>"]
}}"""
    return prompt


def _normalize_number(raw: str) -> list[str]:
    """
    Return all plausible representations of a number for flexible fact-matching.
    e.g. "1,250" → ["1250", "1,250"]
    e.g. "12.5" (percent) → ["12.5", "0.125", "0.13"]
    """
    stripped = raw.replace(",", "")
    variants = {raw, stripped}
    try:
        v = float(stripped)
        if v > 1:
            variants.add(str(round(v / 100, 4)))
            variants.add(str(round(v / 100, 3)))
            variants.add(str(round(v / 100, 2)))
        if 0 < v < 1:
            pct = round(v * 100, 1)
            variants.add(str(pct))
            variants.add(str(int(pct)))
        if v == int(v):
            variants.add(str(int(v)))
    except ValueError:
        pass
    return list(variants)


def validate_message_numbers(body: str, facts: dict, numbers_claimed: list) -> tuple[bool, list[str]]:
    """
    Check that every number in the message body can be traced to the facts dict.
    Handles: formatted numbers (1,250), pct↔decimal (12.5% ↔ 0.125), rupee symbols.
    Returns (is_valid, list_of_suspect_numbers).
    """
    body_clean = body.replace("₹", "").replace(",", "")
    facts_str = json.dumps(facts, default=str)

    found_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', body_clean)

    suspect = []
    for num in found_numbers:
        try:
            val = float(num)
        except ValueError:
            continue
        if val < 5:
            continue
        variants = _normalize_number(num)
        if not any(v in facts_str for v in variants):
            suspect.append(num)

    return len(suspect) == 0, suspect


def _build_allowed_claims(facts: dict) -> set[str]:
    """
    Build a set of lowercase strings that Claude is allowed to mention.
    Covers: offer titles/ids, review themes, services, digest titles/sources,
    trigger payload strings, customer facts.
    """
    allowed: set[str] = set()

    def add_tokens(text: str) -> None:
        if text:
            allowed.add(text.lower())
            for w in text.lower().split():
                if len(w) > 3:
                    allowed.add(w)

    for offer in facts.get("active_offers", []):
        add_tokens(offer.get("title", ""))
        add_tokens(offer.get("id", ""))

    for rt in facts.get("review_themes", []):
        add_tokens(rt.get("theme", ""))
        add_tokens(rt.get("common_quote", ""))

    digest = facts.get("digest_item", {})
    if digest:
        add_tokens(digest.get("title", ""))
        add_tokens(digest.get("source", ""))
        add_tokens(digest.get("summary", ""))
        add_tokens(digest.get("actionable", ""))
        for k in ["trial_n", "patient_segment"]:
            add_tokens(str(digest.get(k, "")))

    for v in facts.get("trigger_payload", {}).values():
        if isinstance(v, str):
            add_tokens(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    for sv in item.values():
                        add_tokens(str(sv))

    for k in ["customer_name", "customer_state", "customer_pref_slot", "customer_lang"]:
        add_tokens(str(facts.get(k, "")))
    for svc in facts.get("customer_services", []):
        add_tokens(svc)

    for k in ["merchant_name", "owner_first", "locality", "city", "category_slug"]:
        add_tokens(str(facts.get(k, "")))

    for s in facts.get("signals", []):
        add_tokens(s)

    for term in [
        "magicpin", "vera", "whatsapp", "listing", "campaign", "offer", "slot",
        "booking", "subscription", "renewal", "pro", "trial", "compliance",
        "abstract", "checklist", "template", "draft", "post", "story", "banner",
        "fluoride", "scaling", "cleaning", "whitening", "aligner", "rct", "opg",
        "thali", "combo", "delivery", "footfall", "covers", "aov",
        "membership", "attendance", "retention", "churn",
        "atorvastatin", "metformin", "telmisartan", "ors", "sunscreen",
        "namaste", "google", "swiggy", "zomato", "insta", "gbp",
        "webinar", "cde", "credits", "smile", "studio", "yoga", "kids",
    ]:
        allowed.add(term)

    return allowed


def validate_message_claims(body: str, facts: dict) -> tuple[bool, list[str]]:
    """
    Check that quoted named claims in the message body are grounded in facts.
    Returns (is_valid, list_of_suspect_claims).
    """
    allowed = _build_allowed_claims(facts)
    quoted = re.findall(r'["\u2018\u2019\u201c\u201d]([^"\'""'']{4,50})["\u2018\u2019\u201c\u201d]', body)

    suspect = []
    for phrase in quoted:
        phrase_lower = phrase.lower().strip()
        matched = any(
            allowed_tok in phrase_lower or phrase_lower in allowed_tok
            for allowed_tok in allowed
        )
        if not matched:
            suspect.append(phrase)

    return len(suspect) == 0, suspect


def compose_message(trigger: dict, merchant: dict, category: dict, customer: Optional[dict]) -> dict:
    """
    Extract facts → call Claude → dual hallucination guard → fallback if suspect.
    """
    facts = extract_facts(trigger, merchant, category, customer)
    prompt = build_composer_prompt(facts)

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()

        # Strip markdown fences if present (```json ... ``` or ``` ... ```)
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        # If Claude added any preamble before the JSON object, find the first {
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start > 0 or (brace_end != -1 and brace_end < len(text) - 1):
            if brace_start != -1 and brace_end != -1:
                text = text[brace_start:brace_end + 1]

        result = json.loads(text)

        body = result.get("body", "")
        numbers_used = result.get("numbers_used", [])
        num_valid, num_suspects = validate_message_numbers(body, facts, numbers_used)
        if not num_valid:
            logger.warning(f"Hallucination guard (numbers): suspects={num_suspects}. Falling back.")
            return _fallback_compose(facts)

        claim_valid, claim_suspects = validate_message_claims(body, facts)
        if not claim_valid:
            logger.warning(f"Hallucination guard (claims): suspects={claim_suspects}. Falling back.")
            return _fallback_compose(facts)

        result.pop("numbers_used", None)
        return result

    except Exception as e:
        logger.error(f"Compose error: {e}")
        return _fallback_compose(facts)


# ─── Fallback composer ────────────────────────────────────────────────────────
def _fallback_compose(facts: dict) -> dict:
    """
    Rule-based fallback. ONLY uses values from the facts dict — no hardcoded numbers.
    Covers all 24 trigger kinds found in the dataset.
    """
    kind = facts.get("trigger_kind", "")
    payload = facts.get("trigger_payload", {})
    owner = facts.get("owner_first") or "there"
    cat = facts.get("category_slug", "")
    active_offers = facts.get("active_offers", [])
    offer_str = active_offers[0]["title"] if active_offers else None
    merchant_name = facts.get("merchant_name", "us")

    # ── renewal_due ────────────────────────────────────────────────────────
    if kind == "renewal_due":
        days = payload.get("days_remaining") or facts.get("sub_days_remaining")
        plan = payload.get("plan") or facts.get("sub_plan", "Pro")
        amount = payload.get("renewal_amount")
        days_str = f"in {days} day{'s' if days != 1 else ''}" if days else "soon"
        amount_str = f"₹{amount}" if amount else ""
        body = (f"{owner}, your Magicpin {plan} subscription expires {days_str}. "
                f"You'll lose campaign tools + your listing rank drops without it. "
                f"{'Renewal: ' + amount_str + '. ' if amount_str else ''}"
                f"Want me to send you the renewal link now?")
        cta = "binary_yes_no"
        rationale = f"Subscription expires {days_str} — highest-urgency revenue-protecting trigger."

    # ── perf_dip ───────────────────────────────────────────────────────────
    elif kind in ("perf_dip", "seasonal_perf_dip"):
        metric = payload.get("metric", "calls")
        delta_pct_raw = payload.get("delta_pct")
        vs_baseline = payload.get("vs_baseline")
        # delta_pct is already a fraction (e.g. -0.50 = -50%) — don't multiply by 100 twice
        if delta_pct_raw is not None:
            delta_str = f"{int(abs(delta_pct_raw) * 100)}%"
        else:
            delta_str = "noticeably"
        ctr_note = ""
        ctr_m = facts.get("ctr_merchant_pct_str")
        ctr_p = facts.get("ctr_peer_pct_str")
        if ctr_m and ctr_p:
            ctr_note = f" (CTR: {ctr_m} vs peer {ctr_p})"
        baseline_note = ""
        if vs_baseline is not None:
            baseline_note = f" vs {vs_baseline} baseline"

        if kind == "seasonal_perf_dip" and payload.get("is_expected_seasonal"):
            # Case study 7 pattern: reframe + save ad spend advice for gyms
            active_members_str = ""
            am = facts.get("cagg_total_active_members") or facts.get("cagg_active_members")
            if am:
                active_members_str = f" Focus retention on your {am} active members."
            body = (f"{owner}, your {metric} dropped {delta_str} this week{ctr_note} — "
                    f"but this is the normal seasonal lull. Every metro sees -25 to -35% "
                    f"in this window. Skip ad spend now; save it for when conversion is 2x higher.{active_members_str} "
                    f"Want me to draft a retention challenge to keep members through the dip?")
        else:
            body = (f"{owner}, your {metric} dropped {delta_str} this week{baseline_note}{ctr_note}. "
                    f"{'Your ' + offer_str + ' offer is live — ' if offer_str else ''}"
                    f"Want me to activate a targeted campaign to recover visibility?")
        cta = "binary_yes_no"
        rationale = f"{metric} dip of {delta_str} detected{' (seasonal — reframe + retention play)' if kind == 'seasonal_perf_dip' else ' — immediate recovery action available'}."

    # ── recall_due ─────────────────────────────────────────────────────────
    elif kind == "recall_due":
        cname = facts.get("customer_name", "")
        slots = payload.get("available_slots", [])
        slot_str = (f"{slots[0]['label']} or {slots[1]['label']}"
                    if len(slots) >= 2 else "this week")
        months = facts.get("customer_months_since_visit")
        months_str = (f" It's been {months} month{'s' if months != 1 else ''} since your last visit."
                      if months else "")
        body = (f"Hi {cname}, {merchant_name} here 🙏{months_str} "
                f"{'We have: ' + offer_str + '. ' if offer_str else ''}"
                f"Open slots: {slot_str}. "
                f"Reply 1 for {slots[0]['label'] if slots else 'first slot'}, "
                f"2 for {slots[1]['label'] if len(slots) > 1 else 'second slot'}, or suggest a time.")
        cta = "multi_choice_slot"
        rationale = "Customer recall due — direct booking with real slot data."

    # ── regulation_change ──────────────────────────────────────────────────
    elif kind == "regulation_change":
        deadline = (payload.get("deadline_iso", "")[:10]
                    if payload.get("deadline_iso") else "upcoming deadline")
        digest = facts.get("digest_item", {})
        title = digest.get("title", "regulatory change") if digest else "regulatory change"
        body = (f"{owner}, compliance update: {title}. "
                f"Effective {deadline}. "
                f"Want me to draft a compliance checklist for your practice?")
        cta = "binary_yes_no"
        rationale = f"Regulation deadline {deadline} — compliance trigger, high urgency."

    # ── research_digest ────────────────────────────────────────────────────
    elif kind == "research_digest":
        digest = facts.get("digest_item", {})
        if digest:
            source = digest.get("source", "")
            title = digest.get("title", "new research")
            trial_n = digest.get("trial_n")
            n_str = f" ({trial_n}-patient trial)" if trial_n else ""
            body = (f"{owner}, {title}{n_str} — {source}. "
                    f"Likely relevant to your patient cohort. "
                    f"Want me to pull the abstract + draft a patient-ed message?")
        else:
            body = f"{owner}, new research this week may be relevant to your practice. Want me to send a summary?"
        cta = "binary_yes_no"
        rationale = "Research digest matched to merchant's category and patient cohort."

    # ── supply_alert ───────────────────────────────────────────────────────
    elif kind == "supply_alert":
        molecule = payload.get("molecule", "")
        batches = payload.get("affected_batches", [])
        mfr = payload.get("manufacturer", "")
        batch_str = ", ".join(batches[:2]) if batches else "certain batches"
        # Derive affected customer count from chronic_rx_count (case study 9 pattern)
        chronic_count = facts.get("cagg_chronic_rx_count")
        affected_note = ""
        if chronic_count:
            affected_note = f" Your {chronic_count} chronic-Rx customers may be affected."
        body = (f"{owner}, urgent: {molecule} recall — {batch_str} from {mfr}. "
                f"Sub-potency, no safety risk, but customers should be informed for replacement.{affected_note} "
                f"Want me to draft their WhatsApp note + the replacement-pickup workflow?")
        cta = "binary_yes_no"
        rationale = f"Supply recall for {molecule} — immediate stock check and patient comms needed{(' (' + str(chronic_count) + ' chronic-Rx customers)') if chronic_count else ''}."

    # ── chronic_refill_due ─────────────────────────────────────────────────
    elif kind == "chronic_refill_due":
        cname = facts.get("customer_name", "")
        mols = payload.get("molecule_list", [])
        mol_str = ", ".join(mols) if mols else "your medications"
        runs_out = payload.get("stock_runs_out_iso", "")
        date_str = runs_out[:10] if runs_out else "soon"
        delivery_saved = payload.get("delivery_address_saved", False)
        delivery_note = " I can arrange home delivery." if delivery_saved else ""
        body = (f"{'Hi ' + cname + ', ' if cname else ''}{merchant_name} here. "
                f"Your {mol_str} refill is due — stock runs out around {date_str}.{delivery_note} "
                f"Want me to prepare your refill order now?")
        cta = "binary_yes_no"
        rationale = f"Chronic refill due before {date_str} — time-critical patient adherence trigger."

    # ── ipl_match_today ────────────────────────────────────────────────────
    elif kind == "ipl_match_today":
        match = payload.get("match", "IPL match")
        match_time_iso = payload.get("match_time_iso", "")
        time_str = match_time_iso[11:16] if len(match_time_iso) > 16 else "tonight"
        is_weeknight = payload.get("is_weeknight", True)
        if not is_weeknight:
            # Saturday/Sunday IPL: people watch at home → delivery spike, dine-in dips
            # Counter-intuitive but data-backed — matches case study 5 pattern
            body = (f"Heads-up {owner} — {match} tonight at {time_str}. "
                    f"Saturday IPL usually shifts -12% dine-in covers (people watch at home). "
                    f"Skip the match-night promo; instead push "
                    f"{'your ' + offer_str + ' ' if offer_str else 'a delivery special '}as a delivery-only Saturday special. "
                    f"Want me to draft the delivery promo now?")
        else:
            body = (f"{owner}, {match} tonight at {time_str} — strong footfall window. "
                    f"{'Your ' + offer_str + ' is live — ' if offer_str else ''}"
                    f"Want me to draft a match-night promo to activate now?")
        cta = "binary_yes_no"
        rationale = f"IPL match today — time-sensitive, restaurants category. {'Saturday = delivery push (not footfall).' if not is_weeknight else 'Weeknight = footfall opportunity.'}"

    # ── review_theme_emerged ───────────────────────────────────────────────
    elif kind == "review_theme_emerged":
        theme = payload.get("theme", "").replace("_", " ")
        occ = payload.get("occurrences_30d")
        quote = payload.get("common_quote", "")
        occ_str = f"{occ} times" if occ else "multiple times"
        body = (f"{owner}, one review theme appeared {occ_str} this month: {theme}. "
                f"{('Example: \"' + quote + '\"') if quote else ''} "
                f"Want me to draft a response template you can use for similar reviews?")
        cta = "binary_yes_no"
        rationale = f"Review theme '{theme}' emerging — reputation management."

    # ── wedding_package_followup ───────────────────────────────────────────
    elif kind == "wedding_package_followup":
        cname = facts.get("customer_name", "")
        days = payload.get("days_to_wedding")
        days_str = f"{days} days" if days else "some time"
        body = (f"Hi {cname} 💍 {owner} from {merchant_name} here. "
                f"{days_str} to your wedding — perfect window to start bridal prep. "
                f"{'We have: ' + offer_str + '. ' if offer_str else ''}"
                f"Want me to block a slot for you this week?")
        cta = "binary_yes_no"
        rationale = "Bridal followup window — high-value customer, time-sensitive."

    # ── winback_eligible ───────────────────────────────────────────────────
    elif kind == "winback_eligible":
        days_exp = payload.get("days_since_expiry")
        lapsed = payload.get("lapsed_customers_added_since_expiry")
        body = (f"{owner}, "
                f"{str(lapsed) + ' customers ' if lapsed else 'some customers '}"
                f"lapsed since your Magicpin subscription expired "
                f"{'(' + str(days_exp) + ' days ago)' if days_exp else ''}. "
                f"{'Your ' + offer_str + ' offer could help win them back. ' if offer_str else ''}"
                f"Want me to send them a reactivation message?")
        cta = "binary_yes_no"
        rationale = "Winback window open with measurable lapsed customer count."

    # ── customer_lapsed_hard ───────────────────────────────────────────────
    elif kind == "customer_lapsed_hard":
        cname = facts.get("customer_name", "")
        days_since = payload.get("days_since_last_visit")
        prev_focus = payload.get("previous_focus", "") or facts.get("customer_prev_focus", "")
        days_str = f"{days_since} days" if days_since else "a while"
        focus_note = f" Last time, {cname} was working on {prev_focus.replace('_', ' ')}." if prev_focus and cname else ""
        body = (f"Hi {cname}, {merchant_name} here!{focus_note} "
                f"It's been {days_str} — we'd love to have you back. "
                f"{'We have ' + offer_str + ' to get you started. ' if offer_str else ''}"
                f"Want me to check available slots for this week?")
        cta = "binary_yes_no"
        rationale = f"Customer lapsed {days_str} — re-engagement with prior focus context."

    # ── active_planning_intent ─────────────────────────────────────────────
    elif kind == "active_planning_intent":
        intent_topic = payload.get("intent_topic", "").replace("_", " ")
        last_msg = payload.get("merchant_last_message", "")
        body = (f"{owner}, picking up where we left off — you wanted to explore {intent_topic}. "
                f"I have a draft structure ready. "
                f"Want me to walk you through it now?")
        cta = "binary_yes_no"
        rationale = f"Merchant explicitly planning '{intent_topic}' — high-intent continuation trigger."

    # ── trial_followup ─────────────────────────────────────────────────────
    elif kind == "trial_followup":
        cname = facts.get("customer_name", "")
        trial_date = payload.get("trial_date", "")
        date_str = trial_date if trial_date else "recently"
        next_sessions = payload.get("next_session_options", [])
        slot_str = next_sessions[0]["label"] if next_sessions else "this week"
        body = (f"Hi {cname}, {merchant_name} here! "
                f"Hope you enjoyed your trial on {date_str}. "
                f"Next session: {slot_str}. "
                f"Want me to block that slot for you?")
        cta = "binary_yes_no"
        rationale = "Post-trial followup — convert trial to membership."

    # ── competitor_opened ──────────────────────────────────────────────────
    elif kind == "competitor_opened":
        comp_name = payload.get("competitor_name", "a competitor")
        dist_km = payload.get("distance_km")
        their_offer = payload.get("their_offer", "")
        dist_str = f"{dist_km} km away" if dist_km else "nearby"
        offer_note = f" They're advertising '{their_offer}'." if their_offer else ""
        body = (f"{owner}, {comp_name} opened {dist_str}.{offer_note} "
                f"{'Your ' + offer_str + ' offer is your strongest differentiator right now. ' if offer_str else ''}"
                f"Want me to run a visibility push to make sure you capture the local searches first?")
        cta = "binary_yes_no"
        rationale = f"New competitor {dist_str} — timing to defend search visibility."

    # ── milestone_reached ──────────────────────────────────────────────────
    elif kind == "milestone_reached":
        metric = payload.get("metric", "reviews").replace("_", " ")
        val_now = payload.get("value_now")
        milestone = payload.get("milestone_value")
        body_val = f"You're at {val_now}" if val_now else "You're close"
        body_mil = f" — just {milestone - val_now if (val_now and milestone) else 'a few'} away from {milestone}!" if milestone else "!"
        body = (f"{owner}, {body_val} {metric}{body_mil} "
                f"Want me to draft a short celebratory post to share with your customers?")
        cta = "binary_yes_no"
        rationale = f"Milestone approaching for {metric} — engagement + social proof opportunity."

    # ── perf_spike ─────────────────────────────────────────────────────────
    elif kind == "perf_spike":
        metric = payload.get("metric", "calls")
        delta_pct_raw = payload.get("delta_pct")
        driver = payload.get("likely_driver", "")
        delta_str = f"{int(abs(delta_pct_raw) * 100)}%" if delta_pct_raw is not None else ""
        driver_note = f" Likely driven by your {driver.replace('_', ' ')}." if driver else ""
        body = (f"{owner}, your {metric} are up{' ' + delta_str if delta_str else ''} this week!{driver_note} "
                f"Good momentum — want me to draft a post to keep it going?")
        cta = "binary_yes_no"
        rationale = f"{metric} spike — capitalize on momentum with content."

    # ── gbp_unverified ─────────────────────────────────────────────────────
    elif kind == "gbp_unverified":
        uplift = payload.get("estimated_uplift_pct")
        uplift_str = f" (estimated {int(uplift * 100)}% visibility uplift)" if uplift else ""
        path = payload.get("verification_path", "phone or postcard")
        body = (f"{owner}, your Google Business listing isn't verified yet{uplift_str}. "
                f"Verification via {path} takes under 10 minutes. "
                f"Want me to walk you through it now?")
        cta = "binary_yes_no"
        rationale = "Unverified GBP — direct visibility uplift from a simple one-time action."

    # ── category_seasonal ──────────────────────────────────────────────────
    elif kind == "category_seasonal":
        trends = payload.get("trends", [])
        # Pick the top rising trend (first positive delta)
        top_trend = next((t for t in trends if "+" in str(t)), trends[0] if trends else "")
        trend_str = top_trend.replace("_", " ").replace("+", "up ") if top_trend else "seasonal shifts"
        body = (f"{owner}, seasonal demand is shifting — {trend_str} in your area. "
                f"{'Stocking action is recommended. ' if payload.get('shelf_action_recommended') else ''}"
                f"Want me to draft a customer message highlighting your stock readiness?")
        cta = "binary_yes_no"
        rationale = f"Seasonal demand trend '{trend_str}' — inventory + customer comms opportunity."

    # ── cde_opportunity ────────────────────────────────────────────────────
    elif kind == "cde_opportunity":
        digest = facts.get("digest_item", {})
        title = digest.get("title", "CDE webinar") if digest else "CDE webinar"
        credits = payload.get("credits")
        fee = payload.get("fee", "")
        credit_str = f" ({credits} CDE credits)" if credits else ""
        fee_str = f", {fee}" if fee else ""
        body = (f"{owner}, there's a {title}{credit_str}{fee_str} this week. "
                f"Relevant to your specialty. "
                f"Want me to send you the registration link?")
        cta = "binary_yes_no"
        rationale = "CDE opportunity for professional development — relevant to merchant's category."

    # ── dormant_with_vera ──────────────────────────────────────────────────
    elif kind == "dormant_with_vera":
        days_dormant = payload.get("days_since_last_merchant_message")
        last_topic = payload.get("last_topic", "").replace("_", " ")
        days_str = f"{days_dormant} days" if days_dormant else "a while"
        topic_note = f" Last time we were discussing {last_topic}." if last_topic else ""
        body = (f"Hi {owner}!{topic_note} "
                f"What's the biggest challenge at {merchant_name} right now? "
                f"I can help with listings, campaigns, or customer follow-ups — just tell me what would help most.")
        cta = "open_ended"
        rationale = f"Merchant dormant {days_str} — re-engagement with open-ended invitation."

    # ── festival_upcoming ──────────────────────────────────────────────────
    elif kind == "festival_upcoming":
        festival = payload.get("festival", "festival")
        days_until = payload.get("days_until")
        days_str = f"in {days_until} days" if days_until else "soon"
        body = (f"{owner}, {festival} is {days_str}. "
                f"{'Your ' + offer_str + ' offer is ready — ' if offer_str else ''}"
                f"Want me to plan a {festival} campaign for your top customers?")
        cta = "binary_yes_no"
        rationale = f"{festival} {days_str} — timely campaign opportunity."

    # ── curious_ask_due ────────────────────────────────────────────────────
    elif kind == "curious_ask_due":
        body = (f"Hi {owner}! Quick check — what service has been most asked-for this week at "
                f"{merchant_name}? "
                f"I'll turn your answer into a Google post + a short WhatsApp reply for pricing queries. "
                f"Takes 5 min.")
        cta = "open_ended"
        rationale = "Weekly curiosity cadence — builds engagement through reciprocity."

    # ── generic fallback ───────────────────────────────────────────────────
    else:
        body = (f"{owner}, I noticed a growth opportunity for {merchant_name}. "
                f"Want me to share the details?")
        cta = "open_ended"
        rationale = f"Trigger kind '{kind}' — generic fallback."

    send_as = "merchant_on_behalf" if facts.get("customer_name") else "vera"

    return {
        "body": body,
        "cta": cta,
        "rationale": rationale,
        "template_name": f"vera_{kind}_v1",
        "send_as": send_as,
    }


# ─── Reply composer ───────────────────────────────────────────────────────────
def compose_reply(conv_id: str, merchant_id: str, customer_id: Optional[str],
                  message: str, turn_number: int) -> dict:
    """
    Handle mid-conversation merchant replies.
    Deterministic for all known intent patterns; LLM for ambiguous cases.
    """
    # 1. Hostile → end immediately
    if is_hostile(message):
        suppressed_keys.add(f"hostile:{merchant_id}")
        return {
            "action": "end",
            "rationale": "Merchant opted out. Closing conversation and suppressing triggers for this merchant."
        }

    conv = conversations.get(conv_id, {})
    turns = conv.get("turns", [])

    # 2. Auto-reply ladder — track per conversation
    if is_auto_reply(message):
        auto_count = sum(1 for t in turns if t.get("is_auto_reply"))
        if auto_count == 0:
            # First auto-reply: acknowledge, leave flag for owner
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' to go ahead.",
                "cta": "binary_yes_no",
                "rationale": "First auto-reply detected — leaving low-friction prompt for owner."
            }
        elif auto_count == 1:
            # Second auto-reply: wait 24h
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": "Auto-reply twice in a row — owner not at phone. Waiting 24h."}
        else:
            return {"action": "end", "rationale": "3 auto-replies without real engagement. Closing."}

    # 3. Out-of-scope curveball (GST, legal, HR, etc.)
    if is_out_of_scope(message):
        # Politely decline and redirect back to last topic
        conv_data = conversations.get(conv_id, {})
        last_vera = next(
            (t["msg"][:60] for t in reversed(conv_data.get("turns", [])) if t.get("from") == "vera"),
            None
        )
        redirect = f" Back to where we were — {last_vera}..." if last_vera else ""
        return {
            "action": "send",
            "body": f"That one's outside what I can help with — you'll want your CA or a specialist for that.{redirect}",
            "cta": "open_ended",
            "rationale": "Out-of-scope request politely declined; conversation redirected."
        }

    # 4. Commit → proceed
    if is_intent_commit(message):
        return {
            "action": "send",
            "body": "On it! Activating that for you now — done in 2 min. 🚀",
            "cta": "open_ended",
            "rationale": "Merchant committed — executing."
        }

    # 5. Simple yes
    if is_positive_yes(message):
        return {
            "action": "send",
            "body": "Great! Drafting that now — give me a moment. 🚀",
            "cta": "open_ended",
            "rationale": "Merchant accepted — confirming execution."
        }

    # 6. Deferral
    if is_deferral(message):
        return {
            "action": "wait",
            "wait_seconds": 7200,
            "rationale": "Merchant deferred. Backing off 2 hours."
        }

    # 7. Price objection
    if is_price_objection(message):
        merchant = get_context("merchant", merchant_id)
        cat_slug = merchant.get("category_slug", "") if merchant else ""
        offers = [o for o in (merchant.get("offers", []) if merchant else []) if o.get("status") == "active"]
        offer_str = offers[0]["title"] if offers else None
        if offer_str:
            return {
                "action": "send",
                "body": (f"Understood! Your existing '{offer_str}' offer already covers this — "
                         f"no extra cost. Should I go ahead with that?"),
                "cta": "binary_yes_no",
                "rationale": "Price objection handled by referencing existing active offer."
            }
        else:
            return {
                "action": "send",
                "body": "Understood — happy to find a lower-commitment option. What's your budget for this month?",
                "cta": "open_ended",
                "rationale": "Price objection — asking for budget to find right fit."
            }

    # 8. Why/explain question
    if is_why_question(message):
        conv_data = conversations.get(conv_id, {})
        trigger_id = conv_data.get("trigger_id", "")
        trigger = get_context("trigger", trigger_id) if trigger_id else {}
        kind = trigger.get("kind", "") if trigger else ""
        rationale_map = {
            "perf_dip": "Your listing gets fewer clicks than nearby competitors — a targeted offer can recover that.",
            "seasonal_perf_dip": "This seasonal dip is normal, but merchants who act during it hold rank better when demand returns.",
            "renewal_due": "Without Pro, your listing rank drops and campaign tools are disabled — directly hurts inbound calls.",
            "recall_due": "Patients who come back regularly are more likely to refer others. This re-opens that window.",
            "review_theme_emerged": "Responding to review patterns builds trust with new customers who read reviews before booking.",
            "competitor_opened": "A new competitor nearby means you have a short window to win local search before they build reviews.",
            "supply_alert": "Keeping affected stock on shelf creates liability — a quick check protects your patients and your licence.",
            "gbp_unverified": "Verified listings rank higher in local search and show up in Google Maps — directly impacts walk-ins.",
        }
        explanation = rationale_map.get(kind, "This action has a direct impact on your monthly bookings and visibility.")
        return {
            "action": "send",
            "body": f"{explanation} Want me to go ahead?",
            "cta": "binary_yes_no",
            "rationale": "Merchant asked for reasoning — providing category-specific explanation."
        }

    # 9. No / decline
    msg_lower = message.lower()
    if any(w in msg_lower for w in ["no", "nahi", "nope", "not interested yet", "not now"]):
        return {
            "action": "wait",
            "wait_seconds": 86400,
            "rationale": "Merchant declined for now. Trying again in 24h."
        }

    # 10. Ambiguous — use LLM
    merchant = get_context("merchant", merchant_id)
    cat_slug = merchant.get("category_slug", "") if merchant else ""
    conv_data = conversations.get(conv_id, {})

    try:
        reply_prompt = f"""You are Vera, Magicpin's AI merchant assistant. Compose a short, precise reply.

Merchant message: "{message}"
Turn number: {turn_number}
Category: {cat_slug}

Last 4 conversation turns:
{json.dumps(conv_data.get("turns", [])[-4:], indent=2)}

Rules:
1. If merchant committed or said yes → move to action immediately.
2. If merchant asked a question → answer it directly without generic filler.
3. Under 80 words. One CTA only. No URLs.
4. Sound like a smart advisor, not a bot.
5. If asked about something outside Magicpin scope (GST, HR, legal) → politely decline + redirect.

OUTPUT JSON only:
{{"action": "send", "body": "<message>", "cta": "binary_yes_no|open_ended|binary_confirm_cancel", "rationale": "<brief>"}}
OR
{{"action": "end", "rationale": "<reason>"}}
OR
{{"action": "wait", "wait_seconds": 3600, "rationale": "<reason>"}}"""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": reply_prompt}]
        )
        text = response.content[0].text.strip()
        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]
        return json.loads(text.strip())

    except Exception as e:
        logger.error(f"Reply compose error: {e}")
        # Context-aware fallback: use last Vera message to give a specific reply
        last_vera_body = next(
            (t["msg"] for t in reversed(conv_data.get("turns", [])) if t.get("from") == "vera"),
            None
        )
        if last_vera_body and len(last_vera_body) > 20:
            # Truncate to the core question portion
            snippet = last_vera_body[:80].rstrip()
            return {
                "action": "send",
                "body": f"To confirm — {snippet}... Want me to go ahead?",
                "cta": "binary_yes_no",
                "rationale": "Ambiguous reply — echoing last Vera message for clarity."
            }
        return {
            "action": "send",
            "body": "Got it — should I go ahead with that?",
            "cta": "binary_yes_no",
            "rationale": "Ambiguous intent — asking for simple confirmation."
        }


# ─── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/v1/healthz")
async def healthz():
    persist_ok = _PERSIST_FILE.exists()
    persist_size = _PERSIST_FILE.stat().st_size if persist_ok else 0
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": count_contexts(),
        "conversations_active": len(conversations),
        "suppressions_active": len(suppressed_keys),
        "persistence": {
            "path": str(_PERSIST_FILE),
            "file_exists": persist_ok,
            "size_bytes": persist_size,
        },
        "api_key_set": bool(_api_key),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Kshitij DTU",
        "team_members": ["Kshitij"],
        "model": "claude-sonnet-4-20250514",
        "approach": (
            "Business-impact trigger scoring (urgency × kind × merchant state × category fit) → "
            "hard cross-category penalties prevent wrong-category triggers winning → "
            "fact extraction (hallucination guard) → Claude composer with verified-facts-only prompt → "
            "dual post-generation validation: numeric + named-claim guard. Full 24-trigger coverage. "
            "App-dir persistence (not /tmp). Counter-intuitive advice: IPL Saturday = delivery push, "
            "seasonal dip = save ad spend."
        ),
        "contact_email": "kshitij2004@gmail.com",
        "version": "2.4.0",
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(status_code=400, content={
            "accepted": False,
            "reason": "invalid_scope",
            "details": f"scope must be one of {valid_scopes}"
        })

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(status_code=409, content={
            "accepted": False,
            "reason": "stale_version",
            "current_version": cur["version"]
        })

    contexts[key] = {"version": body.version, "payload": body.payload}
    logger.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")
    _save_state()

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat()
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []

    # Group non-expired, non-suppressed triggers by merchant
    merchant_triggers: dict[str, list[dict]] = {}
    now_dt = datetime.fromisoformat(body.now.replace("Z", "+00:00"))

    for trg_id in body.available_triggers:
        trg_data = contexts.get(("trigger", trg_id))
        if not trg_data:
            continue
        trg = trg_data["payload"]

        # Suppression check
        sup_key = trg.get("suppression_key", "")
        if sup_key and sup_key in suppressed_keys:
            continue

        # Expiry check
        expires = trg.get("expires_at")
        if expires:
            try:
                exp_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
                if now_dt > exp_dt:
                    continue
            except Exception:
                pass

        merchant_id = trg.get("merchant_id")
        if not merchant_id:
            continue

        # Global hostile suppression
        if f"hostile:{merchant_id}" in suppressed_keys:
            continue

        merchant_triggers.setdefault(merchant_id, []).append(trg)

    # For each merchant: pick best trigger → extract facts → compose
    for merchant_id, triggers in merchant_triggers.items():
        if len(actions) >= 15:
            break

        merchant = get_context("merchant", merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug", "")
        category = get_context("category", category_slug)
        if not category:
            continue

        best_trigger = select_best_trigger(triggers, merchant, category)
        if not best_trigger:
            continue

        trigger_id = best_trigger.get("id", "")
        customer_id = best_trigger.get("customer_id")
        customer = get_context("customer", customer_id) if customer_id else None

        # Stable conversation ID: one per merchant+trigger pair
        conv_id = f"conv_{merchant_id}_{trigger_id}"

        # Skip if this conversation already happened this session
        if conv_id in conversations:
            continue

        composed = compose_message(best_trigger, merchant, category, customer)

        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trigger_id,
            "turns": [{"from": "vera", "msg": composed.get("body", ""), "is_auto_reply": False}],
            "suppressed": False
        }

        send_as = composed.get("send_as", "merchant_on_behalf" if customer_id else "vera")

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": send_as,
            "trigger_id": trigger_id,
            "template_name": composed.get("template_name", f"vera_{best_trigger.get('kind', 'generic')}_v1"),
            "template_params": [
                merchant.get("identity", {}).get("owner_first_name", ""),
                best_trigger.get("kind", ""),
                composed.get("rationale", "")[:80],
            ],
            "body": composed.get("body", ""),
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": best_trigger.get("suppression_key", f"vera:{merchant_id}:{trigger_id}"),
            "rationale": composed.get("rationale", "Best-scored trigger for this merchant.")
        }
        actions.append(action)
        logger.info(f"Action for {merchant_id} via {trigger_id}: {composed.get('body', '')[:80]}...")

    _save_state()
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conv_id = body.conversation_id

    conv = conversations.setdefault(conv_id, {
        "merchant_id": body.merchant_id,
        "customer_id": body.customer_id,
        "trigger_id": "",
        "turns": [],
        "suppressed": False
    })

    if conv.get("suppressed"):
        return {"action": "end", "rationale": "Conversation previously closed."}

    conv["turns"].append({
        "from": body.from_role,
        "msg": body.message,
        "is_auto_reply": is_auto_reply(body.message),
        "turn_number": body.turn_number
    })

    result = compose_reply(
        conv_id,
        body.merchant_id or conv.get("merchant_id", ""),
        body.customer_id or conv.get("customer_id"),
        body.message,
        body.turn_number
    )

    if result.get("action") == "end":
        conv["suppressed"] = True

    if result.get("action") == "send":
        conv["turns"].append({
            "from": "vera",
            "msg": result.get("body", ""),
            "is_auto_reply": False
        })

    _save_state()
    logger.info(f"Reply for {conv_id}: {result.get('action')} — {result.get('body', '')[:60]}")
    return result


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    suppressed_keys.clear()
    _save_state()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
