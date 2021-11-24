from model import get_result  # calling model func
from bottle import Bottle, response, request as bottle_request
import requests  
import json
import urllib

class BotHandlerMixin:  
    BOT_URL = None

    def get_chat_id(self, data):
        """
        Method to extract chat id from telegram request.
        """
        chat_id = data['message']['chat']['id']

        return chat_id

    def get_message(self, data):
        """
        Method to extract message id from telegram request.
        """
        message_text = data['message']['text']

        return message_text

    def send_message(self, prepared_data):
        """
        Prepared data should be json which includes at least `chat_id` and `text`
        """       
        message_url = self.BOT_URL + 'sendMessage'
        requests.post(message_url, json=prepared_data)


class TelegramBot(BotHandlerMixin, Bottle):  
    BOT_URL = 'https://api.telegram.org/bot<TOKEN>/'
    BOT_URL_FILE = 'https://api.telegram.org/file/bot<TOKEN>/'

    def __init__(self, *args, **kwargs):
        super(TelegramBot, self).__init__()
        self.route('/', callback=self.post_handler, method="POST")

    def photo(self, data):
        answer = self.get_json_answer(data, 'Okay, now wait a few seconds!!!')
        self.send_message(answer)

        ide = data['message']['photo'][1]['file_id']
        r = requests.get('{}getFile?file_id={}'.format(self.BOT_URL, ide),)
        
        path_json = json.loads(r.text)
        path = path_json['result']['file_path']

        url = '%s/%s' % (self.BOT_URL_FILE, path)
        file_name = "image.jpg"
        urllib.request.urlretrieve(url,file_name)
        
        return get_result(file_name)

    def prepare_data_for_answer(self, data):
        if 'text' in data['message'] and 'start' in self.get_message(data):
            answer = 'Hi! Send me an image of the animal.'

        elif 'photo' in data['message']:
            answer = self.photo(data)
        else:
            answer = 'Please, send me one image of the animal.'

        return self.get_json_answer(data, answer)

    def get_json_answer(self, data, answer):
        chat_id = self.get_chat_id(data)
        json_data = {
            "chat_id": chat_id,
            "text": answer,
            "parse_mode": "Markdown",
        }

        return json_data

    def post_handler(self):
        data = bottle_request.json
        answer_data = self.prepare_data_for_answer(data)
        self.send_message(answer_data)

        return response


if __name__ == '__main__':
    app = TelegramBot()
    app.run(host='localhost', port=8080)
