from deployment import app


@app.route('/')
def index():
    return 'Welcome to the image processing app'  # Customize the message as needed
