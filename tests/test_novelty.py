import unittest

from autoevolve.novelty import code_similarity


class NoveltyTests(unittest.TestCase):
    def test_similarity_ignores_protected_code_and_cosmetic_edits(self):
        left = (
            "protected = 1\n# EVOLVE-BLOCK-START\nvalue = 2  # old\n"
            "print(value)\n# EVOLVE-BLOCK-END\n"
        )
        right = (
            "protected = 999\n# EVOLVE-BLOCK-START\nvalue=2 # new\n"
            "print(value)\n# EVOLVE-BLOCK-END\n"
        )
        self.assertEqual(code_similarity(left, right), 1.0)

    def test_similarity_detects_substantive_mutable_change(self):
        left = "# EVOLVE-BLOCK-START\nvalue = 2\nprint(value)\n# EVOLVE-BLOCK-END\n"
        right = "# EVOLVE-BLOCK-START\nvalue = 3\nprint(value * 2)\n# EVOLVE-BLOCK-END\n"
        self.assertLess(code_similarity(left, right), 0.995)


if __name__ == "__main__":
    unittest.main()
