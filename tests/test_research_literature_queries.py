"""Live literature queries preserve scientific names and object scope."""
import unittest
from unittest.mock import patch

from farfield.extras.livefeed import (
    ArxivFeed, CompositeFeed, FeedBlocked, object_survey_phrases, survey_queries,
)


class LiteratureQueryTests(unittest.TestCase):
    def test_scientific_names_are_not_stemmed_for_external_search(self):
        phrases = object_survey_phrases("Iris species classification under feature-scale shift")
        self.assertIn("Iris species", phrases)
        self.assertIn("feature-scale shift", phrases)
        self.assertNotIn("iri specie", phrases)

    def test_raw_multiword_terms_and_acronyms_survive(self):
        for topic, phrase in [
            ("Research on SARS-CoV-2 transmission under population shifts", "SARS-CoV-2 transmission"),
            ("Navier-Stokes equations in turbulent flows", "Navier-Stokes equations"),
            ("Finite element analysis with adaptive meshes", "Finite element"),
            ("CRISPR-Cas9 specificity under sequence variation", "CRISPR-Cas9 specificity"),
        ]:
            with self.subTest(topic=topic):
                self.assertIn(phrase, object_survey_phrases(topic))

    def test_single_scientific_name_is_a_valid_object_anchor(self):
        self.assertEqual(object_survey_phrases("Iris"), ("Iris",))
        self.assertEqual(object_survey_phrases("RNA"), ("RNA",))

    def test_full_intent_is_not_repeated_as_an_exact_phrase_constraint(self):
        topic = "Iris species classification under feature-scale shift"
        queries = survey_queries(topic, topic, topic=topic)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertFalse(any(topic in query for query in queries))
        self.assertTrue(any('"Iris species"' in query for query in queries))

    def test_far_mechanism_remains_anchored_to_research_object(self):
        queries = survey_queries("Iris species", "feedback control", topic="Iris species classification")
        for query in queries:
            if "feedback control" in query:
                self.assertIn('all:"Iris species" AND', query)
        self.assertTrue(any("feedback control" in query for query in queries))

    def test_identical_pair_does_not_multiply_requests_without_a_topic(self):
        self.assertEqual(survey_queries("Iris species", "Iris species"), ('all:"Iris species"',))

    def test_composite_query_does_not_force_overlapping_exact_phrases(self):
        topic = "Iris species classification under feature-scale shift"
        queries = []

        def blocked_source(query, *, max_results):
            queries.append(query)
            return FeedBlocked(query, "offline test: no bibliography supplied")

        with patch.object(ArxivFeed, "survey_around", return_value=FeedBlocked("arxiv", "offline")), \
                patch.object(CompositeFeed, "_scholar", side_effect=blocked_source), \
                patch.object(CompositeFeed, "_openalex", side_effect=blocked_source):
            result = CompositeFeed().survey_around(topic, topic, topic=topic)
        self.assertIsInstance(result, FeedBlocked)
        self.assertEqual(len(queries), 2, "identical full intent must not add another query")
        self.assertTrue(all("Iris species" in query for query in queries))
        self.assertFalse(any("species classification" in query for query in queries))


if __name__ == "__main__":
    unittest.main()
