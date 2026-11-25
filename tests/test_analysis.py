import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import OpenAIError
from pydantic import ValidationError

from nativ.analysis import AnalysisError, _enforce_consistency, analyze_request
from nativ.models import Request, RequestAnalysis


class AnalyzeRequestTests(unittest.TestCase):
    def test_blank_request_returns_all_missing_fields_without_calling_llm(self) -> None:
        request = Request(text=" \n ", received_at=datetime.now().astimezone())

        with patch("nativ.analysis._get_client") as get_client:
            result = analyze_request(request)

        get_client.assert_not_called()
        self.assertEqual(
            result,
            RequestAnalysis(
                address=None,
                incident_type=None,
                summary=None,
                urgency=None,
                urgency_reason=None,
                missing_fields=["address", "incident_type", "details"],
            ),
        )

    def test_missing_address_and_type_are_added_to_missing_fields(self) -> None:
        parsed = RequestAnalysis(
            address=None,
            incident_type=None,
            summary="חסר מידע",
            urgency=None,
            urgency_reason=None,
            missing_fields=[],
        )
        parse = Mock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        )
        request = Request(
            text="יש תקלה",
            received_at=datetime.now().astimezone(),
        )

        with patch("nativ.analysis._get_client", return_value=client):
            result = analyze_request(request)

        self.assertEqual(result.missing_fields, ["address", "incident_type"])
        self.assertIs(parse.call_args.kwargs["response_format"], RequestAnalysis)
        self.assertEqual(parse.call_args.kwargs["temperature"], 0)

    def test_provider_error_is_replaced_with_generic_analysis_error(self) -> None:
        parse = Mock(side_effect=OpenAIError("provider details"))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(parse=parse))
        )
        request = Request(
            text="A streetlight is broken",
            received_at=datetime.now().astimezone(),
        )

        with patch("nativ.analysis._get_client", return_value=client):
            with self.assertRaises(AnalysisError) as error:
                analyze_request(request)

        self.assertEqual(
            str(error.exception),
            "ניתוח הפנייה נכשל. נסו שוב מאוחר יותר.",
        )
        self.assertNotIn("provider details", str(error.exception))

    def test_incident_type_is_limited_to_evaluation_categories(self) -> None:
        with self.assertRaises(ValidationError):
            RequestAnalysis(
                address=None,
                incident_type="electrical",
                summary=None,
                urgency=None,
                urgency_reason=None,
                missing_fields=["incident_type"],
            )

    def test_street_without_number_is_treated_as_missing_address(self) -> None:
        result = _enforce_consistency(
            RequestAnalysis(
                address="רחוב הדקל",
                incident_type="water",
                summary="נזילה",
                urgency="medium",
                urgency_reason="נזילה",
                missing_fields=[],
            )
        )

        self.assertIsNone(result.address)
        self.assertEqual(result.missing_fields, ["address"])

    def test_junction_without_number_is_kept(self) -> None:
        analysis = RequestAnalysis(
            address="צומת הדקל והאלון",
            incident_type="roads",
            summary="בור",
            urgency="high",
            urgency_reason="בור פתוח",
            missing_fields=[],
        )

        self.assertEqual(_enforce_consistency(analysis), analysis)

    def test_urgency_is_cleared_when_details_are_missing(self) -> None:
        result = _enforce_consistency(
            RequestAnalysis(
                address="הדקל 3",
                incident_type=None,
                summary="משהו לא תקין",
                urgency="medium",
                urgency_reason="לא ברור",
                missing_fields=["incident_type", "details"],
            )
        )

        self.assertIsNone(result.urgency)
        self.assertIsNone(result.urgency_reason)


if __name__ == "__main__":
    unittest.main()
