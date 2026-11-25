"""Analyze resident requests with Gemini and return structured results."""

import os

import truststore

truststore.inject_into_ssl()

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from nativ.models import Request, RequestAnalysis


_REQUEST_TIMEOUT_SECONDS = 240
_MAX_RETRIES = 0
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_MODEL = "gemini-3.5-flash-lite"
_GENERIC_ERROR_MESSAGE = "ניתוח הפנייה נכשל. נסו שוב מאוחר יותר."

_DEVELOPER_INSTRUCTIONS = """נתח את פניית התושב והחזר את כל השדות לפי סכמת הפלט.

סוגי מפגעים (incident_type):
- lighting: חשמל ותאורה, כגון פנסי רחוב וחוטי חשמל.
- sanitation: תברואה וניקיון, כגון פינוי אשפה, פסולת וגרפיטי.
- roads: כבישים, מדרכות ותנועה, כגון בורות, מדרכות שבורות ורמזורים.
- water: מים, ביוב וניקוז, כגון נזילות, ביוב עולה, הצפות ומכסים.
- parks: גינון ועצים, כגון גינות ציבוריות, דשא ועצים.
- other: כל מה שלא מתאים לקטגוריות האחרות.
אם אי אפשר לזהות את סוג המפגע, החזר null והוסף incident_type ל-missing_fields.

רמות דחיפות (urgency):
- high: סכנה מיידית לבטיחות או לבריאות, כגון חוטי חשמל חשופים, עץ שחוסם כביש,
  ביוב עולה, בור פתוח או רמזור שאינו פועל.
- medium: פגיעה בבטיחות או באיכות החיים בלי סכנה מיידית, כגון פנס כבוי,
  מדרכה שבורה, אשפה שלא פונתה או נזילה.
- low: מטרד, כגון גרפיטי, דשא גבוה, פנס מהבהב או ספסל שבור.
קבע את הדחיפות לפי המפגע עצמו ולא לפי מילות התושב; למשל, "דחוף מאוד" על דשא
גבוה אינו מצדיק urgency גבוה. אם אי אפשר לקבוע דחיפות, החזר null.

שדות חסרים (missing_fields):
- address: אין כתובת מזוהה, הכוללת רחוב ומספר או צומת.
- incident_type: אי אפשר לזהות את סוג המפגע.
- details: אין מספיק פרטים כדי להבין מה הבעיה.

אל תמציא כתובת: החזר רק כתובת או צומת שמופיעים במפורש בפנייה, ואחרת החזר
address כ-null והוסף address ל-missing_fields. אם יש כמה מפגעים, התמקד בחמור
או המרכזי שבהם. כתוב את summary תמיד בעברית. אל תוסיף שדות שאינם בסכמה."""

_FEW_SHOT_MESSAGES: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "פנס הרחוב ברחוב החרוב 7 לא עובד כבר מאתמול.",
    },
    {
        "role": "assistant",
        "content": (
            '{"address":"רחוב החרוב 7","incident_type":"lighting",'
            '"summary":"פנס הרחוב ברחוב החרוב 7 אינו פועל.",'
            '"urgency":"medium",'
            '"urgency_reason":"פנס רחוב כבוי פוגע באיכות החיים אך אינו מתואר '
            'כסכנה מיידית.","missing_fields":[]}'
        ),
    },
    {
        "role": "user",
        "content": "בצומת הרחובות התמר והגפן יש בור פתוח בכביש.",
    },
    {
        "role": "assistant",
        "content": (
            '{"address":"צומת הרחובות התמר והגפן","incident_type":"roads",'
            '"summary":"בור פתוח בכביש בצומת הרחובות התמר והגפן.",'
            '"urgency":"high",'
            '"urgency_reason":"בור פתוח בכביש מהווה סכנה מיידית לבטיחות.",'
            '"missing_fields":[]}'
        ),
    },
    {
        "role": "user",
        "content": "דחוף מאוד, תטפלו בזה!",
    },
    {
        "role": "assistant",
        "content": (
            '{"address":null,"incident_type":null,'
            '"summary":"לא נמסרו פרטים שמאפשרים לזהות את המפגע.",'
            '"urgency":null,"urgency_reason":null,'
            '"missing_fields":["address","incident_type","details"]}'
        ),
    },
]

load_dotenv()

_client: OpenAI | None = None


class AnalysisError(Exception):
    """Raised when a request cannot be analyzed."""


def _get_client() -> OpenAI:
    global _client

    if _client is None:
        try:
            _client = OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url=_GEMINI_BASE_URL,
                timeout=_REQUEST_TIMEOUT_SECONDS,
                max_retries=_MAX_RETRIES,
            )
        except OpenAIError:
            raise AnalysisError(_GENERIC_ERROR_MESSAGE) from None
    return _client


def analyze_request(request: Request) -> RequestAnalysis:
    """Analyze a resident request and return its structured result."""
    if not request.text.strip():
        return RequestAnalysis(
            address=None,
            incident_type=None,
            summary=None,
            urgency=None,
            urgency_reason=None,
            missing_fields=["address", "incident_type", "details"],
        )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "developer", "content": _DEVELOPER_INSTRUCTIONS},
        *_FEW_SHOT_MESSAGES,
        {"role": "user", "content": request.text},
    ]

    try:
        completion = _get_client().chat.completions.parse(
            model=_MODEL,
            messages=messages,
            response_format=RequestAnalysis,
            temperature=0,
        )
    except (OpenAIError, ValidationError, ValueError):
        raise AnalysisError(_GENERIC_ERROR_MESSAGE) from None

    if not completion.choices:
        raise AnalysisError(_GENERIC_ERROR_MESSAGE)

    result = completion.choices[0].message.parsed
    if result is None:
        raise AnalysisError(_GENERIC_ERROR_MESSAGE)

    missing_fields = list(result.missing_fields)
    if result.address is None and "address" not in missing_fields:
        missing_fields.append("address")
    if result.incident_type is None and "incident_type" not in missing_fields:
        missing_fields.append("incident_type")

    if missing_fields != result.missing_fields:
        result = result.model_copy(update={"missing_fields": missing_fields})

    return result
