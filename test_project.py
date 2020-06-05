from unittest import TestCase
import final_project


# nosetests -s --nologcapture  --pdb test_project.py

class TestUtils(TestCase):

    def test_code_to_sign(self):
        pattern = final_project.code_to_sign_pattern(7)
        self.assertEqual(pattern, [1, -1, -1, -1])

        pattern = final_project.code_to_sign_pattern(0)
        self.assertEqual(pattern, [1, 1, 1, 1])

        pattern = final_project.code_to_sign_pattern(11)
        self.assertEqual(pattern, [-1, 1, -1, -1])

        pattern = final_project.code_to_sign_pattern(15)
        self.assertEqual(pattern, [-1, -1, -1, -1])



