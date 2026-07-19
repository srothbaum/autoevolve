import unittest

from autoevolve.patching import PatchError, apply_patch, parse_patch


SOURCE = """# fixed
# EVOLVE-BLOCK-START
value = 1
print(value)
# EVOLVE-BLOCK-END
"""


class PatchingTests(unittest.TestCase):
    def test_applies_exact_patch_inside_evolve_block(self):
        response = """Try a larger value.
<<<<<<< SEARCH
value = 1
=======
value = 2
>>>>>>> REPLACE
"""
        result = apply_patch(SOURCE, response)
        self.assertIn("value = 2", result)
        self.assertNotIn("value = 1", result)

    def test_rejects_ambiguous_search(self):
        source = "# EVOLVE-BLOCK-START\nvalue = 1\nvalue = 1\n# EVOLVE-BLOCK-END\n"
        response = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
        with self.assertRaisesRegex(PatchError, "ambiguous"):
            apply_patch(source, response)

    def test_rejects_change_outside_evolve_block(self):
        response = "<<<<<<< SEARCH\n# fixed\n=======\n# changed\n>>>>>>> REPLACE"
        with self.assertRaisesRegex(PatchError, "outside"):
            apply_patch(SOURCE, response)

    def test_rejects_invalid_python(self):
        response = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue =\n>>>>>>> REPLACE"
        with self.assertRaisesRegex(PatchError, "invalid Python"):
            apply_patch(SOURCE, response)

    def test_rejects_inserted_evolve_markers(self):
        response = (
            "<<<<<<< SEARCH\nvalue = 1\n=======\n"
            "# EVOLVE-BLOCK-END\nvalue = 2\n>>>>>>> REPLACE"
        )
        with self.assertRaisesRegex(PatchError, "EVOLVE markers"):
            apply_patch(SOURCE, response)

    def test_requires_a_patch_block(self):
        with self.assertRaisesRegex(PatchError, "no SEARCH/REPLACE"):
            parse_patch("I have no edit")


if __name__ == "__main__":
    unittest.main()
