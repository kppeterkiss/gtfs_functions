import dotenv

from config import data_root,weather_folder,gtfs_folder,gtfs_stops_file,passanger_info_folder,years_to_process,mav_api_url
import pandas as pd
import requests
import json
from io import StringIO
import re
import datetime
import numpy as np
import os

dotenv.load_dotenv()
owm_key=os.environ['openweathermap_api_key']


'''data_root="data/"
weather_folder=data_root+"odp/"
gtfs_folder=data_root+"gtfsMavMenetrend/"
gtfs_stops_file="stops.txt"
'''
def get_location_data():
    gtfs_stops=pd.read_csv(gtfs_folder+gtfs_stops_file)
    locs=[]
    passanger_info_folders=[passanger_info_folder+str(y)+"/" for y in years_to_process]
    for l in passanger_info_folders:
        passanger_info_locations_file=l+"t_szolg_helyek.txt"
        places_=pd.read_csv(passanger_info_locations_file,sep=',',encoding='iso-8859-2')
        locs.append(places_)
    places=pd.concat(locs,axis=0)
    for l in locs:
        del l
    #TODO check multiple appearence
    places=places.groupby('TELJES_NEV').agg("last")
    places_with_gtfs=places.merge(gtfs_stops,how='left', left_on='POLGARI_NEV', right_on='stop_name')
    return places_with_gtfs


# Egy állomás napi menetrendjének lehívása
def pull_station_data_from_API(station_name):
    station_query = {
        "a": "STATION", "jo": {
            "a": f"{station_name}"
        }
    }
    response = requests.post(mav_api_url, json=station_query)
    return response


def extract_plus_info(df):
    info = []
    columns_to_drop = []
    for c in df.columns:
        if 'Unnamed' in c[0]:
            info.append(c[1])
            full_nan = df[c].isnull().all()
            if full_nan:
                columns_to_drop.append(c)
            print('Full nan col?', c, full_nan)

    return df.drop(columns=columns_to_drop), ' - '.join(list(set(info)))

def pull_train_data_from_API(train_no):
    train_query={
      "a": "TRAIN",
      "jo": {
        "vsz": str(train_no),
        "zoom": True
      }
    }
    response = requests.post(mav_api_url, json=train_query)
    return response

def process_train(erk,ind):
    new_cols_trains = ['ERK_TERV', 'ERK_TENY', 'IND_TERV', 'IND_TENY']
    # ez nem jó, a tegnap indultak közlekedési napja tegnap!!
    # d=datetime.date.today()
    erk_teny,erk_terv,ind_teny,ind_terv=process_plan_fact_time_cols(erk, ind)
    return pd.Series(dict(zip(new_cols_trains,[erk_terv,erk_teny,ind_terv,ind_teny])))
def process_train_desc_t(train_desc):
    train_desc.split()
    # Regular expression to find text within parentheses
    pattern = r'\((.*?)\)'
    # Find all matches
    matches = re.findall(pattern, train_desc)
    train_desc_2 = matches[0]
    train_desc_2_parts=train_desc_2.split(", ")
    date=pd.to_datetime(train_desc_2_parts[1]).date()
    start_station=train_desc_2_parts[0].split(" - ")[0]
    end_station=train_desc_2_parts[0].split(" - ")[1]
    train_desc_1=train_desc.replace("("+train_desc_2+")", "")
    train_desc_1_parts=train_desc_1.split()
    train_no=train_desc_1_parts[0]
    train_name=train_desc_1_parts[-1]
    train_type=' '.join(train_desc_1_parts[1:-1])
    if train_type.isupper():
        t=train_type
        train_type=train_name
        train_name=t
    return date,train_no,train_name,start_station,end_station,train_type

def get_train_data(train_no):
    print('-> Pulling ', train_no)
    # ez kell hogy lássok mi a tény és mi az előrejelzés, kellhet
    #now = datetime.now().time()
    train_no_l = train_no
    if not str(train_no).startswith('55'):
        train_no_l = int('55' + str(train_no))

    train_resp = pull_train_data_from_API(train_no_l)
    train_dict = json.loads(train_resp.text)
    try:
        train_df = pd.read_html(StringIO(train_dict['d']['result']['html']))[0]
    except:
        print('----------------------------- Failed', train_no)
        # failed[train_no] = train_dict
        return None
    train_df, info = extract_plus_info(train_df)
    #print(info)

    #print(train_df.columns)
    train_desc = train_df.columns[0][0]
    date, train_no, train_name, start_station, end_station, train_type = process_train_desc_t(train_desc)
    new_col_dict = {c: c[-1] for c in train_df.columns}
    #print(new_col_dict)
    train_df.columns = new_col_dict.values()
    train_data = pd.concat(
        [train_df, train_df.apply(lambda x: process_train(x["Érk."], x["Ind."]), axis=1)],
        axis=1)
    #print(train_data.columns)
    #print(train_data.index)

    train_data[['ERK_TERV', 'ERK_TENY', 'IND_TERV', 'IND_TENY']] = train_data[['ERK_TERV', 'ERK_TENY', 'IND_TERV', 'IND_TENY']].map(
        lambda x: pd.to_datetime(str(date)+' '+x, format='%Y-%m-%d %H:%M' , errors='coerce') if not pd.isnull(x) else x)
    to_drop = ['Érk.', "Ind."]
    train_data.drop(columns=to_drop, inplace=True)
    train_data['VONAT'] = train_no
    train_data['KOZLEKEDESI_NAP'] = date
    train_data['NEV'] = train_name
    train_data['TiPUS'] = train_type
    train_data['PLUSZ'] = info
    return train_data


def process_plan_fact_time_cols(erk, ind):
    # NaN-nál lehaé
    erk_teny = np.NaN
    erk_terv = np.NaN
    if not pd.isnull(erk):
        erk_a = erk.split()

        erk_terv = erk_a[0]
        if len(erk_a) == 2:
            erk_teny = erk_a[1]
    ind_teny = np.NaN
    ind_terv = np.NaN
    if not pd.isnull(ind):
        ind_a = ind.split()
        ind_terv = ind_a[0]
        if len(ind_a) == 2:
            ind_teny = ind_a[1]
    return erk_teny, erk_terv, ind_teny, ind_terv


def process_train_desc(erk, ind, input):
    new_cols_station = ['KOZLEKEDESI_NAP', 'ERK_TERV', 'ERK_TENY', 'IND_TERV', 'IND_TENY', 'VONATSZAM', 'VONATTIPUS',
                        'IND_VEGALLOMASROL', 'VEGALLOMAS', 'ERK_VEGALLOMASRA']

    d = datetime.date.today()
    input = input.replace(u'\xa0', " ")
    input = " ".join(input.split())
    ia = input.split(" ")
    arrival_time = ia[-1]
    train_no = ia[0]
    i = 1
    train_type = ""
    other_station = ""
    while (not "--" in ia[i]) and (not ":" in ia[i]):
        train_type = ' '.join([train_type, ia[i]])
        i += 1
    start_time = ia[i]
    i = -2
    while (not "--" in ia[i]) and (not ":" in ia[i]):
        other_station = ' '.join([ia[i], other_station])
        i -= 1
    erk_teny, erk_terv, ind_teny, ind_terv = process_plan_fact_time_cols(erk, ind)
    return pd.Series(dict(zip(new_cols_station,
                              [d, erk_terv, erk_teny, ind_terv, ind_teny, train_no, train_type, start_time,
                               other_station, arrival_time])))


# process_train_desc(ind,erk,inp)

def get_station_data(name):
    print(f'Getting data for {name}')
    station_resp = pull_station_data_from_API(name)
    station_dict = json.loads(station_resp.text)
    stat_df = pd.read_html(StringIO(station_dict['d']['result']))[0]
    new_col_dict = {c:c[2] for c in stat_df.columns}
    stat_df.columns= new_col_dict.values()
    station_data=pd.concat([stat_df, stat_df.apply(lambda x: process_train_desc(x["Érk."],x["Ind."],x["Vonat  Viszonylat"]),axis=1)], axis=1)
    station_data[['ERK_TERV','ERK_TENY','IND_TERV','IND_TENY','IND_VEGALLOMASROL','ERK_VEGALLOMASRA']]=station_data[['ERK_TERV','ERK_TENY','IND_TERV','IND_TENY','IND_VEGALLOMASROL','ERK_VEGALLOMASRA']].apply(lambda x: pd.to_datetime(x, format='%H:%M',errors='coerce').dt.time)
    to_drop=['Érk.',"Ind.","Vonat  Viszonylat"]
    station_data.drop(columns=to_drop,inplace=True)
    station_data['ALLOMAS']=name
    #kozlekedesi nap javítása TODO gondoljuk át
    station_data.loc[station_data['ERK_TERV']<station_data['IND_VEGALLOMASROL'],'KOZLEKEDESI_NAP']=station_data['KOZLEKEDESI_NAP']-datetime.timedelta(days=1)
    return station_data

# minden figyelt volnalra, állomásonként lehívjuk a menetrendet, a jelölt vonataink azok lesznek, amik legalább 3 állomáson átmennek
# ezt naponta, mondjuk 00:01-kor lwhúzzuk, hogy tudjuk mi a terv aznapra
def get_trains_on_lines(main_stations):
    trains_per_lines = {}
    for k, v in main_stations.items():
        print(k, v)
        stat_df_s = {}
        for station in v:
            stat_df_s[station] = get_station_data(station)
        station_df = pd.concat(stat_df_s.values(), axis=0)
        vc = station_df['VONATSZAM'].value_counts()
        #melyik vonatok mennek át legalább 3 állomáson
        trains_per_lines[k] = vc[vc > 3].index.to_list()
        #upsert('station_events',station_df,engine)
    return trains_per_lines

def load_location_data():
    return pd.read_pickle(data_root+"stat_coord_dict.pkl")

def load_weather_meta():
    """
    Loads locations of used meteorlogical stations
    Returns:
    pd.Dataframe: name, lat, lon
    """
    weather_meta_file_name = weather_folder + "weather_meta_avg.csv"
    met_stat_locations = pd.read_csv(weather_meta_file_name, sep=',', encoding='iso-8859-2')


def get_daily_weather_forcast(lat=47.969911,lon=21.767344):
    """
       Loads locations of used meteorlogical stations
       Returns:
       pd.Dataframe: name, lat, lon
    """
    url=f"https://api.openweathermap.org/data/2.5/forecast?units=metric&lat={lat}&lon={lon}&appid={owm_key}"
    response = requests.get(url)
    return process_forecast(response.json())

def process_forecast(wf_dict):
    """
       Takes a dict received from OWM, and calculates daily weather forcast
       Returns:
       Dict:
        tx : max temp,
        t: avg temp,
        tn: min temo
        r: precipitation
    """
    txa=[]
    tna=[]
    pa=[]
    for rc in wf_dict['list'][:24]:
        txa.append(float(rc['main']['temp_max']))
        tna.append(float(rc['main']['temp_min']))
        p=0.0
        if float(rc['pop'])>0.0:
            if 'rain' in rc:
                p += float(rc['pop']) * float(rc['rain']['3h'])
            if 'snow' in rc:
                p += float(rc['pop']) * float(rc['snow']['3h'])
        pa.append(p)
        tx=max(txa)
        tn=min(tna)
        p=sum(pa)
    return {'tx':tx,'t':(tx+tn)/2,'tn':tn,'r':p}

#def load_model():
def pull_recents_trains_data_from_API():
    train_query={
      "a": "TRAINS",
      "jo": {
        "history": True,
        "id": True
      }
    }
    response = requests.post(mav_api_url, json=train_query)
    return response


###################################### models loading
import joblib
# modellek mentése, beolvasása:
# könyvtárszerkezet: <model gyökér>/<model típus>/<használt adat típusa>/<plusz infó>/<model file(ok)>

import matplotlib.pyplot as plt

import os

model_location = "model/"
# Check if the directory exists
if not os.path.exists(model_location):
    # Create the directory
    os.makedirs(model_location)


def get_location_and_name(model_name, model_desc, data_desc):
    loc = model_location + '/' + model_name + '/' + data_desc + '/' + model_desc + "/"
    name = model_name + '_' + data_desc + '_' + model_desc
    return loc, name


def save_sklearn_model(model, model_name, data_desc, model_desc='base'):
    loc, name = get_location_and_name(model_name, model_desc, data_desc)
    if not os.path.exists(loc):
        # Create the directory
        os.makedirs(loc)
    # Save the model to a file
    joblib.dump(model, f'{loc}{name}.joblib')


def get_nn_model_name(epochs, data_desc):
    return f'NN_{data_desc}_epoch_{epochs}'


import torch

def save_NN(model, X_scaler, y_scaler, history, data_desc, model_desc='base'):
    loc, name = get_location_and_name('NN', model_desc, data_desc)
    if not os.path.exists(loc):
        # Create the directory
        os.makedirs(loc)
    save_sklearn_model(X_scaler, 'X_scaler', data_desc)
    save_sklearn_model(y_scaler, 'y_scaler', data_desc)
    torch.save(model, loc + name + '.pth')

    for k, v in history.items():
        plt.plot(v, label='lr=' + str(k))

    plt.legend(loc="upper left")

    plt.savefig(loc + 'history.png')


def load_NN(path):
    ret = {}
    for f in os.listdir(path):
        if f.endswith('.pth'):
            print('Loading model:', path + f)
            ret['model'] = torch.load(path + f, weights_only=False)
        elif f.endswith('.joblib'):
            if 'X_scaler' in f:
                ret['X_scaler'] = joblib.load(path + f)
            elif 'y_scaler' in f:
                ret['y_scaler'] = joblib.load(path + f)
            else:
                print(f'Unspecified joblib file: {f}')
        else:
            print(f'Unspecified file: {f}')
    return ret


def load_NNs(model_path):
    # könyvtárszerkezet: <model gyökér>/<model típus>/<használt adat típusa>/<plusz infó>/<model file(ok)>

    ret = {}
    # dir=data_desc
    for dir in os.listdir(model_path + 'NN'):
        ret[dir] = {}
        if os.path.isdir(model_path + 'NN/' + dir + '/'):
            print('reading ' + dir)
            # dir2=model_desc
            for dir2 in os.listdir(dir):
                print('reading ' + dir + '/' + dir2)
                ret[dir][dir2]['model'] = load_NN(model_path + 'NN/' + dir + '/' + dir2)
                ret[dir][dir2]['data_desc'] = dir
                ret[dir][dir2]['model_desc'] = dir2
    return ret


def load_NN_by_desc(model_desc, data_desc):
    loc, name = get_location_and_name('NN', model_desc, data_desc)
    ret = load_NN(loc)
    ret['model_desc'] = model_desc
    ret['data_desc'] = data_desc
    return ret


def load_sk_by_desc(model_name, model_desc, data_desc):
    loc, name = get_location_and_name(model_name, model_desc, data_desc)
    return {'model': joblib.load(f'{loc}/{name}.joblib'), 'model_name': model_name, 'data_desc': data_desc,
            'model_desc': model_desc}


# kelleni fog: 'MENETREND_IDO (m)','ELOZO_SZAKASZ_KESES (m)','KESES (m)','tx','t','tn','r','TERV_IDOTARTAM (m)'
def convert_real_time_to_ml_data(input_df, weather_series):
    date_format = '%Y-%m-%d %H:%M:%S'

    # input_df['ERK_TERV'].fillna(input_df['IND_TERV'], inplace=True)
    # input_df['IND_TERV'].fillna(input_df['ERK_TERV'], inplace=True)

    # először szétválasztjuk az eseményeket
    departures = input_df[['Állomás', 'IND_TERV', 'IND_TENY', 'KOZLEKEDESI_NAP']]
    departures.rename(columns={'IND_TERV': 'IDO', 'IND_TENY': 'TENY_IDO'}, inplace=True)

    arrivals = input_df[['Állomás', 'ERK_TERV', 'ERK_TENY', 'KOZLEKEDESI_NAP']]

    arrivals.rename(columns={'ERK_TERV': 'IDO', 'ERK_TENY': 'TENY_IDO'}, inplace=True)
    ml_df = pd.concat([arrivals, departures]).sort_values(by='IDO')
    ml_df = ml_df.dropna(axis=0, subset=['IDO'], thresh=1)

    ml_df['OSSZ_KESES'] = ml_df['TENY_IDO'] - ml_df['IDO']
    ml_df[['ELOZO_OSSZ_KESES', 'ELOZO_ALLOMAS', 'ELOZO_ESEMENY_IDO']] = ml_df[['OSSZ_KESES', 'Állomás', 'IDO']].shift()
    # ahol nincsen tényidő és nincs előrejelzés, ott kell számolnunk.
    ml_df['ELOZO_ESEMENY_IDO'].fillna(ml_df['IDO'], inplace=True)

    ml_df['KESES'] = ml_df['OSSZ_KESES'] - ml_df['ELOZO_OSSZ_KESES']
    ml_df['KESES'].fillna(ml_df['OSSZ_KESES'], inplace=True)

    ml_df['OSSZ_KESES (m)'] = ml_df['OSSZ_KESES'] / np.timedelta64(1, 'm')
    ml_df['TERV_IDOTARTAM (m)'] = (ml_df['IDO'] - ml_df['ELOZO_ESEMENY_IDO']).dt.seconds / 60
    ml_df['KESES (m)'] = ml_df['KESES'] / np.timedelta64(1, 'm')
    ml_df['MENETREND_IDO (m)'] = ml_df['IDO'].dt.hour * 60 + ml_df['IDO'].dt.minute

    ml_df.drop(columns=['KESES', 'OSSZ_KESES', 'ELOZO_ESEMENY_IDO'])

    ml_df['TENY_IDOTARTAM (m)'] = ml_df['TERV_IDOTARTAM (m)'] + ml_df['KESES (m)']
    ml_df[['ELOZO_SZAKASZ_TERV_IDOTARTAM (m)', 'ELOZO_SZAKASZ_KESES (m)']] = ml_df[
        ['TERV_IDOTARTAM (m)', 'KESES (m)']].shift()

    series_repeated = pd.concat([weather_series] * len(ml_df), axis=1).T.reset_index(drop=True)
    print(series_repeated)
    series_repeated.index = ml_df.index
    # Append the series to the DataFrame
    ml_df = pd.concat([ml_df, series_repeated], axis=1)

    return ml_df

def iterative_prediction(iterative_pred_df,model):
    last_pred=None
    preds=[]
    #iterative_pred_df['pred']=np.NaN
    for i in iterative_pred_df.index:
        if last_pred:
            iterative_pred_df.at[i,'ELOZO_SZAKASZ_KESES (m)']=last_pred
        r = iterative_pred_df.loc[[i]]
        #print('last_p', last_pred)
        #rdf=pd.DataFrame(r).T
        #print(rdf)
        p=model.predict(r)
        #print(p)
        preds.append(p[0])
        last_pred=p[0]
    iterative_pred_df['pred']=preds
import pickle
from config import gtfs_dowload_location
def load_geom_dbs():
    file = open(data_root+'gtfs_shapes.pkl', 'rb')
    mapping = pickle.load(file)
    shapes=pd.read_csv(gtfs_dowload_location + "latest/gtfs/shapes.txt")
    mapping_with_shapes=mapping.merge(shapes, how='left')
    mapping_with_shapes=mapping_with_shapes[['VONATSZAM','shape_pt_lat','shape_pt_lon','shape_pt_sequence']]
    return mapping_with_shapes

import geojson
def get_geometry(train_no, mapping_df):
    geom_act = mapping_df[mapping_df['VONATSZAM'] == train_no]
    if geom_act.empty():
        print(f'No geometry for {train_no}')
        return None
    points = list(zip(geom_act['shape_pt_lon'], geom_act['shape_pt_lat']))
    # Define coordinates for the LineString
    coordinates = points

    # Create a LineString object
    line = geojson.LineString(coordinates)

    # Convert the LineString object to a JSON string
    # line_json = geojson.dumps(line)
    return line



def transform_recent_trains(recent_trains):
    recent_trains=recent_trains[['@Delay','@TrainNumber','@Lat','@Lon','VONAT_','NEV_first','@Relation','KOZLEKEDESI_NAP_last']]
    recent_trains['Vonatszam']=recent_trains['VONAT_'].astype(int)
    recent_trains['Nev']=recent_trains['NEV_first']+' : '+recent_trains['@Relation']
    recent_trains['Nap']=recent_trains['KOZLEKEDESI_NAP_last'].astype(str)
    recent_trains=recent_trains.rename(columns={'@Lat':'Lat','@Lon':'Lon','@Delay':'Keses'})
    return recent_trains[['Vonatszam','Lat','Lon','Nev','Nap','Keses']]
import json
def get_recent_trains(train_schedules):
    resp = pull_recents_trains_data_from_API().json()
    ts = resp['d']['result']['@CreationTime']
    print(ts)
    trains_json = resp['d']['result']['Trains']['Train']
    current_locations = pd.DataFrame(trains_json)
    current_trains=current_locations.merge(train_schedules, right_on="VONATSZAM_L",left_on="@TrainNumber",how='inner')
    current_trains=transform_recent_trains(current_trains)
    resp={"Timestamp":ts,"trains":current_trains.to_dict('records')}
    return current_trains,json.dumps(resp)

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        '''
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        '''
        if isinstance(obj, datetime.time):
            return str(obj)
        if isinstance(obj, datetime.date):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return str(obj)
        if isinstance(obj,geojson.LineString):
            return geojson.dumps(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.datetime64):
            return str(pd.to_datetime(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from config import main_stations,collected_trains,collected_trains,data_root,weather_folder
import pandas as pd


def get_geometry(train_no, mapping_df):
    geom_act = mapping_df[mapping_df['VONATSZAM'] == train_no]
    points = list(zip(geom_act['shape_pt_lon'], geom_act['shape_pt_lat']))
    # Define coordinates for the LineString
    coordinates = points

    # Create a LineString object
    line = geojson.LineString(coordinates)

    # Convert the LineString object to a JSON string
    # line_json = geojson.dumps(line)
    return line
def init_schedule(main_stations_dict,collected_trains_dict):
    # mi az amit követünk, illetve mi az ami tényleg közlekedik a vonalon
    trains_to_watch = get_trains_on_lines(main_stations_dict)
    for k,v in collected_trains_dict.items():
        trains_to_watch[k] = list(set(map(int, trains_to_watch[k])).intersection(collected_trains_dict[k]))
    # figyelt vonatok végleges listája
    watched_trains_list=[]
    for k,v in trains_to_watch.items():
        watched_trains_list+=v
    all_watched_train_data={}
    for train_no in watched_trains_list:
        all_watched_train_data[train_no]=get_train_data(train_no)
    train_schedules=pd.concat( [v.groupby('VONAT').agg({'IND_TERV': ['first'], 'ERK_TERV': ['last'], 'KOZLEKEDESI_NAP': ['last'],'Állomás':['first','last'],'NEV':'first'}).reset_index() for v in all_watched_train_data.values()])
    train_schedules.columns=train_schedules.columns.map('_'.join)
    train_schedules["VONATSZAM_L"]=train_schedules['VONAT_'].astype(str).apply(lambda x:'55'+x)
    return train_schedules

def add_weather_data(train_schedules,coords,met_stat_locations):
    train_schedules=train_schedules.merge(coords,how='left', left_on=('Állomás_first'),right_on='POLGARI_NEV')
    train_schedules=train_schedules.merge(met_stat_locations,left_on='Legközelebbi met. állomás',right_on='Loc')
    query_data=train_schedules[['Loc','Lat','Lon']].drop_duplicates()
    query_data[['tx','t','tn','r']]=query_data.apply(lambda x: pd.Series(get_daily_weather_forcast(x['Lat'],x['Lon'])),axis=1)
    train_schedules=train_schedules.merge(query_data)
    return train_schedules

def init_data():

    coords=pd.read_pickle(data_root+"stat_coord_dict.pkl")
    weather_meta_file_name=weather_folder+"weather_meta_avg.csv"
    met_stat_locations=pd.read_csv(weather_meta_file_name,sep=',',encoding='iso-8859-2')

    train_schedules=init_schedule(main_stations_dict=main_stations,collected_trains_dict=collected_trains)
    train_schedules=add_weather_data(train_schedules,coords,met_stat_locations)
    return train_schedules,met_stat_locations,coords

def transform_recent_trains(recent_trains):
    recent_trains=recent_trains[['@Delay','@TrainNumber','@Lat','@Lon','VONAT_','NEV_first','@Relation','KOZLEKEDESI_NAP_last']]
    recent_trains['Vonatszam']=recent_trains['VONAT_'].astype(int)
    recent_trains['Nev']=recent_trains['NEV_first']+' : '+recent_trains['@Relation']
    recent_trains['Nap']=recent_trains['KOZLEKEDESI_NAP_last'].astype(str)
    recent_trains=recent_trains.rename(columns={'@Lat':'Lat','@Lon':'Lon','@Delay':'Keses'})
    return recent_trains[['Vonatszam','Lat','Lon','Nev','Nap','Keses']]
import json
def get_recent_trains(train_schedules):
    resp = pull_recents_trains_data_from_API().json()
    ts = resp['d']['result']['@CreationTime']
    #print(ts)
    trains_json = resp['d']['result']['Trains']['Train']
    current_locations = pd.DataFrame(trains_json)
    current_trains=current_locations.merge(train_schedules, right_on="VONATSZAM_L",left_on="@TrainNumber",how='inner')
    current_trains=transform_recent_trains(current_trains)
    resp={"Timestamp":ts,"trains":current_trains.to_dict('records')}
    return current_trains,json.dumps(resp)
def create_train_obj(train_no,lat, lon,name,day,  data_df,delay,type,plus_info='',mapping_with_shapes=None, schedule=False,geom=False):
    train={}
    train['Vonatszam']=train_no
    train['Lat']=lat
    train['Lon']=lon
    train['Nev']=name
    train['Nap']=day
    train['Keses']=delay
    train['TiPUS'] =type
    train['PLUSZ']=plus_info
    if schedule:
        df_copy=data_df.copy()
        df_copy = df_copy.fillna(np.nan).replace([np.nan], [None])
        train['Table']=df_copy.to_dict(orient='records')
    if geom:
        pattern = r"^55"
        replacement = ""
        train_no_1 = int(re.sub(pattern, replacement, str(train_no)))
        train['geom']=get_geometry(train_no_1,mapping_with_shapes)
    return train


def get_recent_train_details(no, train_schedules, model, recent_trains,mapping_with_shapes):
    short_no = no
    if not str(no).startswith('55'):
        short_no = no
        no = int('55' + str(no))
    else:
        short_no = int(str(no).replace('55', ''))
    data = get_train_data(no)
    # nem kellene crashelnie
    sch = train_schedules[train_schedules['VONATSZAM_L'].astype(int) == no].iloc[0]
    weather = sch[['tx', 't', 'tn', 'r']]
    # töröljük az előjelzést

    data[['ERK_TENY', 'IND_TENY']] = data[['ERK_TENY', 'IND_TENY']].map(
        lambda x: pd.NaT if x > datetime.datetime.now() else x)
    data_sv = data[['Km', 'Állomás', 'TiPUS', 'NEV']]
    ml_data = convert_real_time_to_ml_data(data, weather)
    # csak az indulásnál lesz elvileg -TODO randa
    ml_data.loc[0, 'ELOZO_SZAKASZ_KESES (m)'] = 0.0
    td = ml_data.copy()
    past_data = td[~pd.isnull(td['KESES (m)'])]
    past_data['pred'] = model.predict(
        past_data[['MENETREND_IDO (m)', 'ELOZO_SZAKASZ_KESES (m)', 'tx', 't', 'tn', 'r', 'TERV_IDOTARTAM (m)']])
    future_data = td[pd.isnull(td['KESES (m)'])]
    #print('Future - ', future_data.shape)
    iterative_pred_df = future_data[
        ['MENETREND_IDO (m)', 'ELOZO_SZAKASZ_KESES (m)', 'tx', 't', 'tn', 'r', 'TERV_IDOTARTAM (m)']]
    iterative_prediction(iterative_pred_df, model)
    future_data[['ELOZO_SZAKASZ_KESES (m)', 'pred']] = iterative_pred_df[['ELOZO_SZAKASZ_KESES (m)', 'pred']]

    past_data['Last prediction'] = past_data['IDO'] + pd.to_timedelta(past_data['pred'], unit='m')
    future_data['cum_pred'] = future_data['pred'].cumsum()
    future_data['Last prediction'] = future_data['IDO'] + pd.to_timedelta(future_data['cum_pred'], unit='m')
    future_data = future_data.drop(columns=['pred', 'cum_pred'])
    past_data = past_data.drop(columns=['pred'])
    predicted_df = pd.concat([past_data, future_data])
    print(predicted_df.columns)
    ret_df = predicted_df.groupby('Állomás').agg(
        ERK_TERV=('IDO', 'first'),
        ERK_TENY=('TENY_IDO', 'first'),
        IND_TERV=('IDO', 'last'),
        IND_TENY=('TENY_IDO', 'last'),
        Utolso_erk_pred=('Last prediction', 'first'),
        Utolso_ind_pred=('Last prediction', 'last')
    ).reset_index()
    final = ret_df.sort_values(by='ERK_TERV', ascending=True).merge(data_sv)
    #print(short_no)
    #print(recent_trains['Vonatszam'].unique())
    ans = recent_trains.query(f' Vonatszam == {short_no}' ).head(1)
    if not ans.empty:
        lat=ans['Lat'].values[0]
        lon=ans['Lon'].values[0]
        delay = ans['Keses'].values[0]
        day=ans['Nap'].values[0]
        name=ans['Nev'].values[0]
    else:
        lat = None
        lon = None
        delay = None
        day = None
        name = None
    tipus=final['TiPUS'].values[0]
    resp = {'Timestamp': str(datetime.datetime.now()), 'trains': []}
    resp['trains'].append(create_train_obj(train_no=no, lat=lat, lon=lon, name=name, day=day, data_df=final, delay=delay,type=tipus,
                                      mapping_with_shapes=mapping_with_shapes, schedule=True, geom=True))

    return json.dumps(resp, cls=JSONEncoder)


def predict_history(model, ml_data, train_no, train_name,mapping_with_shapes):
    # TODO nem kell folymatosan megjósolni a past_data-t elég egyszer! valami update kellene majd
    ml_data.loc[0, 'ELOZO_SZAKASZ_KESES (m)'] = 0.0
    # alapértelmezett előjelzés - vég tényidő
    last_row = ml_data.tail(1)
    termination_fact = last_row['TENY_IDO'].values[0]
    termonation_schedule = last_row['IDO'].values[0]
    td = ml_data.copy()
    past_data = td[~pd.isnull(td['ELOZO_SZAKASZ_KESES (m)'])]
    # alapértelmezett előjelzés - jósolt késés
    last_fact_row = past_data.tail(1)
    current_delay = last_fact_row['TENY_IDO'].values[0] - last_fact_row['IDO'].values[0]
    arrival_time_def_pred = termonation_schedule + current_delay
    future_data = td[pd.isnull(td['ELOZO_SZAKASZ_KESES (m)'])]

    to_drop = ['VONATSZAM', 'IDO', "TENY_IDO", "ESEMENY_SORSZAM", "RELATIV_KESES", "ID", "OSSZ_KESES (m)",
               'TENY_IDOTARTAM (m)', "Kezdés", "Befejezés", 'KOZLEKEDESI_NAP', 'POLGARI_NEV', 'SZH_NEV', 'stop_name',
               'KESES (m)', 'SZH_KOD_ind', 'stop_lat_ind', 'stop_lon_ind', 'SZH_NEV_ind', 'SZH_KOD_cél', 'stop_lat_cél',
               'stop_lon_cél', 'SZH_NEV_cél', 'cél_szh', 'ind_szh']

    not_ml_data_past = past_data[to_drop]
    not_ml_data_future = future_data[to_drop]
    last_section = not_ml_data_past.tail(1)

    ts = last_section["TENY_IDO"].values[0]
    delay = (last_section["TENY_IDO"] - last_section["IDO"]).dt.total_seconds() / 60
    delay = delay.values[0]
    day = last_section["KOZLEKEDESI_NAP"].dt.date.values[0]
    # if not not_ml_data_future.empty:

    active_section = not_ml_data_future.head(1)
    # print(active_section)
    valid = active_section["TENY_IDO"].values[0]
    lat1 = active_section['stop_lat_cél'].values[0]
    lat2 = active_section['stop_lat_ind'].values[0]
    lon1 = active_section['stop_lon_cél'].values[0]
    lon2 = active_section['stop_lon_ind'].values[0]

    lat = (lat1 + lat2) / 2
    lon = (lon1 + lon2) / 2

    # raw_data[['MENETREND_IDO','ELOZO_KESES','ELOZO VONAT KESES','KESES']] = raw_data[['MENETREND_IDO','ELOZO_KESES','ELOZO VONAT KESES','KESES']].map(convert_time_to_minute)
    past_data.drop(to_drop, axis=1, inplace=True)
    future_data.drop(to_drop, axis=1, inplace=True)
    past_data['pred'] = model.predict(
        past_data)
    # print('Future - ', future_data.shape)
    iterative_pred_df = future_data
    iterative_prediction(iterative_pred_df, model)
    future_data = pd.concat([not_ml_data_future, future_data], axis=1)
    past_data = pd.concat([not_ml_data_past, past_data], axis=1)
    future_data[['ELOZO_SZAKASZ_KESES (m)', 'pred']] = iterative_pred_df[['ELOZO_SZAKASZ_KESES (m)', 'pred']]

    past_data['Last prediction'] = past_data['IDO'] + pd.to_timedelta(past_data['pred'], unit='m')
    future_data['cum_pred'] = future_data['pred'].cumsum()
    future_data['Last prediction'] = future_data['IDO'] + pd.to_timedelta(future_data['cum_pred'], unit='m')
    future_data = future_data.drop(columns=['pred', 'cum_pred'])
    past_data = past_data.drop(columns=['pred'])
    predicted_df = pd.concat([past_data, future_data])
    predicted_termination = predicted_df.tail(1)['Last prediction'].values[0]
    error_of_prediction = abs((predicted_termination - termination_fact).astype('timedelta64[m]'))
    error_of_default_prediction = abs((arrival_time_def_pred - termination_fact).astype('timedelta64[m]'))
    plus_info = f'Alap. ej. hiba: {str(error_of_default_prediction)}, AI ej. hiba: {str(error_of_prediction)},placeholder...: {str(error_of_prediction)}'
    ret_df = predicted_df.groupby('SZH_NEV_cél').agg(
        ERK_TERV=('IDO', 'first'),
        ERK_TENY=('TENY_IDO', 'first'),
        IND_TERV=('IDO', 'last'),
        IND_TENY=('TENY_IDO', 'last'),
        Utolso_erk_pred=('Last prediction', 'first'),
        Utolso_ind_pred=('Last prediction', 'last')
    ).reset_index()
    ret_df.rename(columns={'SZH_NEV_cél': 'Állomás'}, inplace=True)
    ret_df['Km'] = 0
    ret_df['ERK_TERV'] = ret_df['ERK_TERV'].dt.time
    ret_df['ERK_TENY'] = ret_df['ERK_TENY'].dt.time
    ret_df['IND_TERV'] = ret_df['IND_TERV'].dt.time
    ret_df['IND_TENY'] = ret_df['IND_TENY'].dt.time
    ret_df['Utolso_erk_pred'] = ret_df['Utolso_erk_pred'].dt.time
    ret_df['Utolso_ind_pred'] = ret_df['Utolso_ind_pred'].dt.time
    # final = ret_df.sort_values(by='ERK_TERV', ascending=True).merge(data_sv)
    train_obj = create_train_obj(train_no=train_no, lat=lat, lon=lon, name=train_name, day=day, data_df=ret_df,
                                 delay=delay, type="személyvonat",plus_info=plus_info, mapping_with_shapes=mapping_with_shapes,
                                 schedule=True, geom=True)
    # resp = {'Timestamp': str(datetime.datetime.now()), 'trains': []}
    # resp['trains'].append(train_obj)

    return predicted_df, ts, valid, train_obj, error_of_prediction


import scipy.stats as stats


def get_histrory(query_date, gtfs_name_mapping, gtfs_geom_mapping, raw_data, model):
    resp = {"states": []}
    correlations = {}
    state_store = {}
    da = query_date.split('-')
    inteteresting_trains = raw_data[
        (raw_data['IDO'] > datetime.datetime(year=int(da[0]), month=int(da[1]), day=int(da[2]))) & (
                    raw_data['IDO'] < datetime.datetime(year=int(da[0]), month=int(da[1]),
                                                        day=int(da[2])) + pd.Timedelta(days=1))][
        ['KOZLEKEDESI_NAP', 'VONATSZAM']].drop_duplicates()
    daily_df = inteteresting_trains.merge(raw_data, how='left')

    # TODO groupby szeparálni
    groups = daily_df.groupby(["KOZLEKEDESI_NAP", "VONATSZAM"])
    train_nos = {}
    for g_name, single_test_train in groups:
        train_no = int(g_name[1])
        print(train_no)
        try:
            name = gtfs_name_mapping[gtfs_name_mapping['VONATSZAM'] == train_no]['NEV'].iloc[0]
        except:
            name = train_no
        # print(f'missing train name {train_no}')
        train_nos[train_no] = name

        # single_train_keys = pd.DataFrame(inteteresting_trains.iloc[0]).T

        # single_train_keys['KOZLEKEDESI_NAP'] = pd.to_datetime(single_train_keys['KOZLEKEDESI_NAP'])
        # single_test_train = single_train_keys.merge(daily_df, how='left')

        c = gtfs_geom_mapping[['SZH_KOD', 'stop_lat', 'stop_lon', 'SZH_NEV']]
        single_test_train[['ind_szh', 'cél_szh']] = single_test_train['ID'].str.split('-', expand=True)
        single_test_train[['ind_szh', 'cél_szh']] = single_test_train[['ind_szh', 'cél_szh']].astype(int)

        single_test_train = single_test_train.merge(c.add_suffix('_ind'), left_on='ind_szh',
                                                    right_on="SZH_KOD_ind").merge(c.add_suffix('_cél'),
                                                                                  left_on='cél_szh',
                                                                                  right_on="SZH_KOD_cél")
        single_test_train[['stop_lon_ind', 'stop_lat_ind', 'stop_lon_cél', 'stop_lat_cél']] = single_test_train[
            ['stop_lon_ind', 'stop_lat_ind', 'stop_lon_cél', 'stop_lat_cél']].interpolate()
        # TODO mehet kiljebb
        single_test_train['VONAL_STATUSZ_VALTOZOTT'] = single_test_train['VONAL_STATUSZ_VALTOZOTT'].astype(float)
        single_test_train.fillna(0.0, inplace=True)
        errors = []

        for i in single_test_train.index:
            d = single_test_train.copy()
            d.loc[i:, 'ELOZO_SZAKASZ_KESES (m)'] = np.NaN

            # print(name)

            pdf, ts, valid, train_obj, err = predict_history(model, d, train_no, name,gtfs_name_mapping)
            errors.append(err)
            state_store[(train_no, ts, valid)] = train_obj
            del d

        consecutive_numbers = list(range(1, len(errors) + 1))
        correlation, p_value = stats.spearmanr(errors, consecutive_numbers)
        correlations[train_no] = (correlation, p_value)

    return state_store, correlations, train_nos

import copy
def get_historic_trains(date_str, raw_data, mapping_with_shapes, coords, model):
    state_store, corr, train_nos = get_histrory(date_str, mapping_with_shapes, coords, raw_data, model)
    keys = state_store.keys()
    sc = np.mean([c[0] for c in corr.values()])
    p = np.mean([c[1] for c in corr.values()])
    ts_s = [s for n, s, e in keys]
    ts_s.sort()
    # states - ezeken belül
    resp = {"Info": f"SC={sc}, p={p}"}
    all_trains = []

    def copy_small_data(t_o):
        small_o = copy.deepcopy(t_o)
        del small_o['Table']
        del small_o['geom']
        return small_o

    # intervallumonként berakjuk minden vonat megfelelő állapotát
    for first_element, second_element in zip(ts_s, ts_s[1:]):
        state_obj = {}
        state_obj["Timestamp"] = first_element
        state_obj["Valid"] = second_element
        state_obj['trains'] = []
        # print(first_element, second_element)

        idx = 0
        for k, v in state_store.items():
            # if idx>3: continue
            # print(first_element,k[1],second_element)
            if (first_element <= k[1]) and (second_element > k[1]):
                print('here')
                state_obj['trains'].append(copy_small_data(v))
            idx += 1
        all_trains.append(state_obj)
    resp['states'] = all_trains
    details = {}
    for k, v in state_store.items():
        if k[0] not in details:
            details[k[0]] = []
        train_state_obj = {"Timestamp": k[1], 'Valid': k[2], "trains": [v]}
        details[k[0]].append(train_state_obj)
    return json.dumps(resp, cls=JSONEncoder), details


def get_historic_train_details(date_str, train_no, details):
    return json.dumps({"states": details[int(train_no)]}, cls=JSONEncoder)


