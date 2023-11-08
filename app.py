from main import Chat
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
app.config['DEBUG'] = True
chat_instance = Chat(from_disk=True)


@app.route("/")
def index():
    return render_template('index.html', chat_history=chat_instance.chat_history)


@app.route("/submit_question",  methods=["POST"])
def submit_question_handler():
    if request.method == "POST":
        try:
            question = request.form["question"]
            answer = chat_instance.retrieval_answer(question)
            print(answer)
        except Exception as e:
            print("An error occurred: " + str(e), "error")

    return redirect(url_for('index'))


@app.route("/clear_chat",  methods=["POST"])
def clear_chat_handler():
    if request.method == "POST":
        try:
            chat_instance.clear_chat_history()
        except Exception as e:
            print("An error occurred: " + str(e), "error")

    return redirect(url_for('index'))
