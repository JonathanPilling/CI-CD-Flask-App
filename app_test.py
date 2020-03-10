import app
import unittest
import json

class TestPredict(unittest.TestCase):
    
    def setUp(self):
        self.app = app.app.test_client()
        self.app.testing = True

    # Checks response code for GET request #
    def test_get_code(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    # Checks response code for sample POST request #
    def test_post_code(self):
        info = {'Pclass':3, 'Age':2, 'SibSp': 1, 'Fare':50}
        response = self.app.post(
            '/',
            data = json.dumps(info),
            content_type = 'application/json')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
