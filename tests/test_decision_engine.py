import unittest

from src.decision_engine import Outcome, decide


class DecisionEngineTests(unittest.TestCase):
    def test_missing_defect_model_never_accepts_clean_image(self):
        result = decide(image_suitable=True)
        self.assertEqual(result.outcome, Outcome.REVIEW)

    def test_unsuitable_image_requires_recapture(self):
        result = decide(image_suitable=False, validated_policy_passed=True)
        self.assertEqual(result.outcome, Outcome.RECAPTURE)

    def test_supported_defect_causes_rejection(self):
        result = decide(image_suitable=True, supported_defect_detected=True)
        self.assertEqual(result.outcome, Outcome.FAIL)

    def test_ai_only_pass_recommendation_stays_review(self):
        result = decide(image_suitable=True, ai_recommendation="pass")
        self.assertEqual(result.outcome, Outcome.REVIEW)

    def test_only_explicit_validated_policy_can_accept(self):
        result = decide(image_suitable=True, validated_policy_passed=True)
        self.assertEqual(result.outcome, Outcome.PASS)

    def test_ai_warning_keeps_validated_pass_candidate_in_review(self):
        result = decide(image_suitable=True, validated_policy_passed=True, ai_recommendation="review")
        self.assertEqual(result.outcome, Outcome.REVIEW)

    def test_uncertain_category_blocks_defect_verdict_and_acceptance(self):
        result = decide(
            image_suitable=True,
            category_confident=False,
            supported_defect_detected=True,
            validated_policy_passed=True,
        )
        self.assertEqual(result.outcome, Outcome.REVIEW)
        self.assertIn("category is unknown", result.explanation)


if __name__ == "__main__":
    unittest.main()