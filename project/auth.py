from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from . import db
from flask_login import login_user, login_required, logout_user


from keras.preprocessing import image
from PIL import Image
from IPython.core.display import display
import numpy as np
import io
import tensorflow as tf

auth = Blueprint('auth', __name__)

classes = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']

def load_model():
    model = tf.keras.models.load_model('project/my_model.h5')
    return model

model = load_model()

def prepare_image(img, target):

    if img.mode != "RGB":
        img = image.convert("RGB")

    img = img.resize(target)
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return img

@auth.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        email = request.form.get('email')
        print("email", email)
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            response = jsonify({"success": False})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            display(image)
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224)).astype('float32')/255
            display(image, (224,224))
            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            
            data["predictions"] = []
            
            # loop over the results and add them to the list of
            # returned predictions
            for index in range(len(classes)):
                r = {"label": classes[index], "probability": float(preds[0][index]*100)}
                data["predictions"].append(r)
            
            data["predictions"] = sorted(data["predictions"], key=lambda d: d['probability'], reverse=True)[:5]
            # # indicate that the request was a success
            data["success"] = True
            data["result"] = classes[np.argmax(preds)]
    return jsonify(data)


@auth.route('/login')
def login():
    return render_template('login.html')

@auth.route('/login', methods=['POST'])
def login_post():
    # login code goes here
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()

    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    print(email)
    print(user)
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        response = jsonify({"success": False})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        # return redirect(url_for('auth.login')) # if the user doesn't exist or password is wrong, reload the page

    # if the above check passes, then we know the user has the right credentials
    login_user(user, remember=remember)
    response = jsonify({"success": True})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    # return redirect(url_for('main.profile'))

@auth.route('/signup')
def signup():
    return render_template('signup.html')

@auth.route('/signup', methods=['POST'])
def signup_post():
    # code to validate and add user to database goes here
    print(request)
    email = request.form.get('email')
    print(email)
    name = request.form.get('name')
    print(name)
    password = request.form.get('password')
    print(password)

    user = User.query.filter_by(email=email).first() # if this returns a user, then the email already exists in database

    if user: # if a user is found, we want to redirect back to signup page so user can try again
        flash('Email address already exists')
        response = jsonify({"success": False})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        # return redirect(url_for('auth.signup'))

    # create a new user with the form data. Hash the password so the plaintext version isn't saved.
    new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()
    # return redirect(url_for('auth.login'))
    response = jsonify({"success": True})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))