import unittest
from unittest.mock import MagicMock
from gsm8k import GSM8K_evaluation  # Replace with your actual module name
from utils.strings import extract_answer  # Ensure this function is correct

class TestGSM8KEvaluation(unittest.TestCase):
    def setUp(self):
        self.evaluator = GSM8K_evaluation()
        self.evaluator.test = [
            {'question': 'What is 2+2?', 'answer': 'The answer is #### 4'},
            {'question': 'What is 5+3?', 'answer': 'The answer is #### 8'},
            {'question': 'What is 7-2?', 'answer': 'The answer is #### 5'}
        ]

    def mock_model_correct(self, question):
        """Mock model that returns correct answers."""
        answers = {
            'What is 2+2?': 'The answer is #### 4',
            'What is 5+3?': 'The answer is #### 8',
            'What is 7-2?': 'The answer is #### 5'
        }
        return answers.get(question, 'Unknown')

    def mock_model_incorrect(self, question):
        """Mock model that returns incorrect answers."""
        return "The answer is 999."

    def test_eval_correct_model(self):
        """Test evaluation with a model that always returns correct answers."""
        accuracy, results = self.evaluator.eval(self.mock_model_correct)
        self.assertEqual(accuracy, 1.0, "All answers should be correct.")
        self.assertTrue(all(results), "All results should be True.")

    def test_eval_incorrect_model(self):
        """Test evaluation with a model that always returns incorrect answers."""
        accuracy, results = self.evaluator.eval(self.mock_model_incorrect)
        self.assertEqual(accuracy, 0.0, "All answers should be incorrect.")
        self.assertTrue(not any(results), "All results should be False.")

if __name__ == '__main__':
    unittest.main()
