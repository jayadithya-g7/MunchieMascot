import csv
from uuid import uuid4
import pandas as pd
from flask import Flask, request, Response
from hashlib import sha256
from flask_cors import CORS, cross_origin

DATA_PATH = "./server/recipes.csv"
USERS_PATH = "./server/users.csv"

recipe_data = pd.read_csv(DATA_PATH)
# # Ask the user for input food name
# food_name = input("Enter a food name: ").lower()

# # Search for recipes containing the entered food name
# matching_recipes = recipe_data[recipe_data["name"].str.lower(
# ).str.contains(food_name, na=False)]

# if not matching_recipes.empty:
#     # Display the first matching recipe
#     display_recipe(matching_recipes.iloc[0])
# else:
#     print("No recipes found for the entered food name.")


def add_user(username, password):
    with open(USERS_PATH, mode='a', newline='') as users_file:
        users_file_writer = csv.writer(users_file)

        users_file_writer.writerow(
            [uuid4(), username, password, sha256(f'{username}|{password}'.encode()).hexdigest()])


def get_user_from_hash(userhash):
    with open(USERS_PATH, mode='r', newline='') as users_file:
        users_file_reader = csv.reader(users_file)

        for u in users_file_reader:
            if u[-1] == userhash:
                return u
        else:
            return None


def get_recipe_by_id(id):
    return recipe_data[recipe_data["id"].astype(int).eq(id)].to_dict('records')


def find_recipes_by_name(param):
    return recipe_data[recipe_data["name"].str.lower().str.contains(param.lower(), na=False)][["id", "name"]].to_dict('records')


app = Flask(__name__)
cors = CORS(app, resource={
    r"/*": {
        "origins": "*"
    }
})


@app.route('/signup', methods=["POST"])
@cross_origin()
def sign_up():
    print(request)
    username, password = request.form.get(
        'username', None), request.form.get('password', None)
    if not username or not password:
        return Response(status=400)
    add_user(username, password)
    return Response(status=200)


@app.route('/user', methods=["GET"])
@cross_origin()
def check_user_hash():
    userhash = request.args.get('userhash', None)
    if not userhash:
        return Response(status=400)
    user = get_user_from_hash(userhash)
    if not user:
        return Response(status=400)
    return {'id': user[0], 'username': user[1]}


@app.route('/recipe/search', methods=['GET'])
@cross_origin()
def search_recipes():
    userhash = request.args.get('userhash', None)
    q = request.args.get('q', None)
    limit = request.args.get('limit', 10)
    if not q or not userhash:
        return Response(status=400)
    if not get_user_from_hash(userhash):
        return Response(status=401)
    return find_recipes_by_name(q)[:limit]


@app.route('/recipe/<int:recipe_id>', methods=['GET'])
@cross_origin()
def get_recipe(recipe_id: int):
    userhash = request.args.get('userhash', None)

    if not recipe_id or not userhash:
        return Response(status=400)
    if not get_user_from_hash(userhash):
        return Response(status=401)
    recipes = get_recipe_by_id(recipe_id)
    if not recipes or len(recipes) == 0:
        return Response(status=400)
    recipe = recipes[0]
    return {str(i): recipe[i] if '[' not in str(recipe[i]) else eval(recipe[i]) for i in recipe}


if __name__ == '__main__':
    app.run(debug=True)
