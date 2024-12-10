from flask import Flask, jsonify
from flask import request
from flask import send_from_directory
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

from utils import init_data, get_recent_trains, convert_real_time_to_ml_data, iterative_prediction, JSONEncoder, \
    load_geom_dbs, load_sk_by_desc, get_train_data, get_geometry, get_recent_train_details, get_historic_trains,get_historic_train_details
from config import data_root
import pandas as pd
import json
pd.options.mode.chained_assignment = None




app = Flask(__name__, static_folder=os.path.abspath('/vke/'))


train_schedules=None
met_stat_locations=None
coords=None

recent_trains=None
mapping_with_shapes=None
model = None
base_RT =None


raw_data=None
historic_train_details=None


# Define a route for the root URI



@app.route('/<path:path>')
def send_report(path):
    # Using request args for path will expose you to directory traversal attacks
    return send_from_directory('vke', path)


'''
@app.route('/<path:filename>')
def serve_static(filename):
    print('here')
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory('vke', 'index.html')
'''


@app.route('/')
def hello_world():
    try:
        # return send_file('relAdmin/login.html')
        return send_from_directory('vke', 'index.html')
    except Exception as e:
        print(e.args[0])


# Define a route for a specific URI
@app.route('/api/recent_data', methods=['GET'])
def get_data():
    d = request.args.get('Vonatszam')
    if d is None:
        global recent_trains
        recent_trains, resp = get_recent_trains(train_schedules)
        return resp
    else:
        return get_recent_train_details(d, train_schedules, model, recent_trains,mapping_with_shapes)

    return jsonify({"date": str(d)})


# Define another route for a different URI
@app.route('/api/historic_data', methods=['GET'])
def get_message():
    global gtfs_name_mapping,gtfs_geom_mapping,raw_data,base_RT,historic_train_details

    d = request.args.get('date')
    tn=request.args.get('vonatszam')
    print(tn)
    print(d)
    if tn is None:
        resp, det=get_historic_trains(d, raw_data, mapping_with_shapes, coords, base_RT)

        historic_train_details=resp
        return resp
    else:
        return get_historic_train_details(d,tn,historic_train_details)





appHasRunBefore:bool = False

@app.before_request
def firstRun():
    print('first call')
    global appHasRunBefore
    if not appHasRunBefore:
        # Run any of your code here
        global train_schedules
        global met_stat_location
        global coords
        global recent_trains
        global mapping_with_shapes
        global model
        global raw_data
        global base_RT
        print('init ')

        train_schedules, met_stat_locations, coords = init_data()
        recent_trains = None
        mapping_with_shapes = load_geom_dbs()
        model = load_sk_by_desc('RT', 'base', 'Small')['model']
        base_RT = load_sk_by_desc('RT', 'base', 'Base')['model']
        raw_data = pd.read_pickle(data_root + 'data.pkl')
        print('init done')

        # Set the bool to True so this method isn't called again
        appHasRunBefore = True

if __name__ == '__main__':
    app.run(debug=True)
