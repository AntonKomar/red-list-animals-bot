# red-list-animals-bot

The bot identifies animal’s species by photo and defines if it is an endangered kind or not.

The model uses tensorflow and keras for animals prediction.

Also the project requires to install Bottle with ```pip install bottle``` or download the source package at [PyPI](https://pypi.org/project/bottle/).

To test this bot you can host it locally by using ngrok. It allows to expose a web server running on the local machine to the internet. You simply need to tell ngrok what port the web server is listening on.

Afterwards you can set a webhook  to test the bot in the telegram provided by ngrok in the telegram server by running this URL: https://api.telegram.org/bot<TOKEN>/setWebHook?url=https://xxxxxxxxx.ngrok.io
