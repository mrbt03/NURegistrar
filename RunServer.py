from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load models
small_model = joblib.load(r"C:\Users\marko\Downloads\small_model.pkl")
medium_model = joblib.load(r'C:\Users\marko\Downloads\medium_model.pkl')
large_model = joblib.load(r'C:\Users\marko\Downloads\large_model.pkl')

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
# Apply custom term standardizations
custom_replacements = {
    'engg': 'engineering',
    'eng': 'engineering',
    'engineer': 'engineering',
    'engineerng':'engineering',
    'sci': 'science',
    'chem': 'chemistry',
    'biochem': 'biochemistry',
    'biol': 'biology',
    'econ': 'economics',
    'psy': 'psychology',
    'psych': 'psychology',
    'calc': 'calculus',
    'maths': 'mathematics',
    'math': 'mathematics',
    'phys': 'physics',
    'info': 'information',
    'comp': 'computer',
    'cs': 'computerscience',
    'ee': 'electricalengineering',
    'me': 'mechanicalengineering',
    'ce': 'civilengineering',
    'env': 'environmental',
    'stat': 'statistics',
    'stats': 'statistics',
    'biostat': 'biostatistics',
    'neuro': 'neuroscience',
    'phil': 'philosophy',
    'aero': 'aeronautical',
    'aerospace': 'aeronautical',
    'fin': 'finance',
    'acct': 'accounting',
    'mktg': 'marketing',
    'adm': 'administration',
    'admin': 'administration',
    'mgmt': 'management',
    'manag': 'management',
    'int': 'international',
    'dev': 'development',
    'soc': 'sociology',
    'socy': 'sociology',
    'anthro': 'anthropology',
    'geo': 'geography',
    'geog': 'geography',
    'hist': 'history',
    'liter': 'literature',
    'edu': 'education',
    'englit': 'englishliterature',
    'engl': 'english',
    'ling': 'linguistics',
    'cog': 'cognitive',
    'cogn': 'cognitive',
    'behav': 'behavioral',
    'beh': 'behavioral',
    'pol': 'political',
    'poli': 'political',
    'pls': 'political science',
    'sys': 'systems',
    'biotech': 'biotechnology',
    'tech': 'technology',
    'thermo': 'thermodynamic',
    'quant': 'quantitative',
    'clin': 'clinical',
    'diag': 'diagnostics',
    'diagno': 'diagnostics',
    'med': 'medical',
    'medic': 'medical',
    'pharma': 'pharmacology',
    'pharm': 'pharmacology',
    'viro': 'virology',
    'immuno': 'immunology',
    'patho': 'pathology',
    'epi': 'epidemiology',
    'epidemio': 'epidemiology',
    'nano': 'nanotechnology',
    'org': 'organic',
    'inorg': 'inorganic',
    'metab': 'metabolism',
    'metabo': 'metabolism',
    'gen': 'genetics',
    'genet': 'genetics',
    'cancer': 'oncology',
    'onco': 'oncology',
    'cardio': 'cardiology',
    'neurosci': 'neuroscience',
    'ophth': 'ophthalmology',
    'opthal': 'ophthalmology',
    'derm': 'dermatology',
    'derma': 'dermatology',
    'gastro': 'gastroenterology',
    'ortho': 'orthopedics',
    'orthop': 'orthopedics',
    'ped': 'pediatrics',
    'peds': 'pediatrics',
    'psycho': 'psychology',
    'psychia': 'psychiatry',
    'psychiat': 'psychiatry',
    'anat': 'anatomy',
    'anato': 'anatomy',
    'molec': 'molecular',
    'mach': 'machine',
    'learn': 'learning',
    'algo': 'algorithm',
    'algos': 'algorithm',
    'net': 'network',
    'nets': 'networks',
    'oper': 'operations',
    'stud': 'studies',
    'equil': 'equilibrium',
    'adv': 'advanced',
    'intro': 'introduction',
    'mech': 'mechanical',
    'topic': 'topics',
    'computer': 'computing',
    'fluid': 'fluids',
    'mechanic': 'mechanics',
    'physics': 'physical',
    'mat': 'materials',
    'manuf': 'manufacturing',
    'seminars': 'seminar',
    'selected': 'selection',
    'sem': 'seminar',
    'rcr': 'ResponsibleCondunctResearch',
    'training': 'train',
    'proc': 'process',
    'principle': 'principles',
    'statistic': 'statistics',
    'statistical': 'statistics',
    'model': 'modeling',
    'fundamental': 'fundamentals',
    'advanced': 'advance',
    'sim': 'simulation',
    'dynamic': 'dynamics',
    'probability': 'probabilities',
    'envr': 'environmental',
    'civil': 'civil',
    'quantitative': 'quantify',
    'senior': 'senior',
    'mathematics': 'mathematical',
    'intg': 'integration',
    'foundation': 'foundational',
    'learning': 'learn',
    'data': 'data',
    'opt': 'optimization',
    'spec': 'special',
    'struct': 'structure',
    'biomedical': 'biomedicine',
    'management': 'manage',
    'meth': 'method',
    'transport': 'transportation',
    'lab': 'laboratory',
    'biomed': 'biomedicine',
    'technology': 'technological',
    'solid': 'solids',
    'entrepreneurship': 'entrepreneurial',
    'computational': 'compute',
    'biomechanics': 'biomechanical',
    'mod': 'model',
    'mechan': 'mechanics',
    'eq': 'equation',
    'composite': 'composites',
    'development': 'develop',
    'environ': 'environmental',
    'theory': 'theoretical',
    'phase': 'phases',
    'analytics': 'analytical',
    'syst': 'systems',
    'finance': 'financial',
    'scheduling': 'schedule',
    'circuit': 'circuits',
    'hon': 'honor',
    'experimentl': 'experimental',
    'capstone': 'capstone',
    'programming': 'program',
    'biotechnology': 'biotechnological',
    'heat': 'heating',
    'projectmanagement': 'projectmanagement',
    'social': 'society',
    'prog': 'program',
    'mechatronics': 'mechatronic',
    'algorithm': 'algorithmic',
    'datascience': 'datascience',
    'electronics': 'electronic',
    'ie': 'industrialengineering',
    'analy':'analysis',
    'stoch': 'stochastic',
    'thermodyn':'thermodynamic',
    'thermodynamics':'thermodynamic',
    'dsgn':'design',
    'fld': 'fluid',
    'engin':'engineering',
    'num': 'number',
    'tran': 'transport',
    'digita':'digital',
    'bme': 'biomedical engineering',
    'prod':'production',
    'artif': 'artifical',
    'qual':'qualitative',
    'optic':'optical',
    'engrg':'engineering',
    'ai': 'artifical intellegence',
    'struc':'structure',
    'ml':'machine learning',
    'polym':'polymer',
    'dynami':'dynamics',
    'dyn':'dynamics',
    'modelling':'modeling',
    'anal':'analysis',
    'envir':'enviromental',
    'prob':'probabilities',
    'optim':'optimization',
    'movemt':'movement',
    'princ':'principal',
    'mathematical':'mathematics',
    'optimize':'optimization',
    'probabilistic':'probabilities',
    'procss':'process',
    'measuremnt':'measurement',
    'fundmtl':'fundamentals',
    'fundmtls':'fundamentals',
    'hierarch':'hierarchal',
    'physic':'physics',
    'imag':'image',
    'acctg':'accounting',
    'procss':'process',
    'introductory':'introduction',
    'sciences':'science',
    'analysi':'analysis',
    'theromodyn':'thermodynamic',
    'cataly':'catalyst',
    'softwr':'software',
    'biolog':'biology',
    'prin': 'principle',
    'optimizatn': 'optimization',
    'electromag': 'electromagnetic',
    'dynam': 'dynamic',
    'transfrm':'transform',
    'hlthcare':'healthcare',
    'microproc':'microprocessor',
    'algorithmics':'algorithm',
    'acquisitn':'acquisition',
    'civ':'civil',
    'intermed':'intermediate',
    'appld':'applied',
    'electromagnet':'electromagnetic',
    'bio':'biology',
    'contrl':'control',
    'algorithmic':'algorithm',
    'viscoelast':'viscoelastic',
    'electrochem':'electrochemistry',
    'inno':'innovation',
    'mechanistic':'mechanical',
    'biochemical':'biochemistry',
    'inelastic':'elasticity',
    'ctrl':'control',
    'mathematical':'mathematics',
    'algorithmic':'algorithm',
    'biomechcs':'biomechanics',
    'thermal':'heat',
    'manager':'management',
    'principles':'principle',
    'det':'deterministic',
    'bldg':'building',
    'construc':'construction',
    'constr':'construction',
    'strat':'strategy',
    'mgnt':'management',
    'dig':'digital',
    'log':'logistics',
    'dialog':'discourse',
    'exp':'experience',
    'hydrogeol':'hydrogeology',
    'researc': 'research',
    'devel':'develop',
    'netwks':'networks',
    'ops':'operations',
    'prj':'project',
    'methd':'method',
    'transp':'transportation',
    'desgn':'design',
    'mgt':'management',
    'sig':'signal',
    'methd':'method',
    'proj':'project',
    'comm':'communication',
    'appl':'applied',
    'tpcs':'topics',
    'nanosci':'nanoscience',
    'professnls':'professionals',
    'transit':'transportation',
    'scal':'scale',
    'mvmt':'movement',
    'transitio':'transition',
    'financing':'finance',
    'probl':'problem',
    'solvrs':'solvers',
    'mngmnt':'management',
    'hetr':'heterogenous',
    'solvrs':'solvers',
    'mngmnt':'management',
    'hetr':'heterogenous',
    'dif':'differential',
    'sy':'system',
    'compu':'computation',
    'equ':'equation',
    'lang':'language',
    'dsgng':'design',
    'agnt':'agent',
    'paral':'parallel',
    'equat':'equation',
    'progr':'program',
    'lrn':'learn',
    'interac':'interaction',
    'rndm':'random',
    'dvlp':'develop',
    'negot':'negotiations',
    'biom':'biomedical',
    'crit':'critical',
    'diff':'differential',
    'prop':'properties',
    'ldrshp':'leadership',
    'trn':'transportation',
    'interdiscpl':'interdisciplinary',
    'interdispl': 'interdisciplinary',
    'busines':'business',
    'algorythmics':'algorithms',
    'lngs':'languages',
    'fnct': 'function',
    'biomats':'biomaterial',
    'soln':'solution',
    'reg':'regulation',    
    'dvlptmt':'develop',
    'technol':'technology',
    'advanvced':'advanced',
    'sustainabi': 'sustainability',
    'energ':'energy',
    'projec':'project',
    'cntr':'control',
    'fndmntls':'fundamentals',
    'regul':'regulation',
    'dsign':'design',
    'procesng':'processing',
    'managmnt':'management',
    'databs': 'database',
    'ana': 'analysis',
    'entrepren': 'entrepreneurship',
    'artifical': 'artificial',
    'hierarchal': 'hierarchical',
    'intr': 'introduction',
    'agricul': 'agriculture',
    'heterogenous': 'heterogeneous',
    'electroni': 'electronics',
    'markts': 'markets',
    'fundemental': 'fundamental',
    'pertrb': 'perturb',
    'innovat': 'innovation',
    'innov': 'innovation',
    'verif': 'verification',
    'elec': 'electrical',
    'crytalline': 'crystalline',
    'graphcs': 'graphics',

}

def replace_custom_words(tokens):
    return [custom_replacements.get(word, word) for word in tokens]

# Bigram Replacements
bigram_replacements = {
    "engineering analysis": "engineeringanalysis",
    "design thinking": "designthinking",
    "linear algebra": "linearalgebra",
    "stochastic models": "stochasticmodels",
    "material science": "materialscience",
    "computer science": "computerscience",
    "computer engineering": "computerengineering",
    "industrial engineering": "industrialengineering",
    "mechanical engineering": "mechanicalengineering",
    "electrical engineering": "electricalengineering",
    "industrial engineering": "industriallengineering",
    "civil engineering": "civilengineering",
    "chemical engineering": "chemicalengineering",
    "biomedical engineering": "biomedicalengineering",
    "applied mathematics": "appliedmathematics",
    "molecular biology": "molecularbiology",
    "organic chemistry": "organicchemistry",
    "inorganic chemistry": "inorganicchemistry",
    "physical chemistry": "physicalchemistry",
    "data science": "datascience",
    "data engineering":"dataengineering",
    "machine learning": "machinelearning",
    "artificial intelligence": "artificialintelligence",
    "operations research": "operationsresearch",
    "environmental science": "environmentalscience",
    "management science": "managementscience",
    "natural science": "naturalscience",
    "earth science": "earthscience",
    "material properties": "materialproperties",
    "financial engineering": "financialengineering",
    "systems engineering": "systemsengineering",
    "quantitative methods": "quantitativemethods",
    "quantitative analysis": "quantitativeanalysis",
    "human resources": "humanresources",
    "project management": "projectmanagement",
    "supply chain": "supplychain",
    "social science": "socialscience",
    "political science": "politicalscience",
    "game theory": "gametheory",
    "signal processing": "signalprocessing",
    "information technology": "informationtechnology",
    "network security": "networksecurity",
    "software engineering": "softwareengineering",
    "health informatics": "healthinformatics",
    "public health": "publichealth",
    "health science": "healthscience",
    "health care":"healthcare",
    "life science": "lifescience",
    "decision making": "decisionmaking",
    "cognitive science": "cognitivescience",
    "human computer": "humancomputer",
    "cloud computing": "cloudcomputing",
    "decision science": "decisionscience",
    "deep learning": "deeplearning",
    "machine learning": "machinelearning",
    "reinforcement learning": "reinforcementlearning",
    "real time": "realtime",
    "time series": "timeseries",
    "genetic engineering": "geneticengineering",
    "control systems": "controlsystems",
    "electronic devices": "electronicdevices",
    "power systems": "powersystems",
    "digital media": "digitalmedia",
    "critical thinking": "criticalthinking",
    "human rights": "humanrights",
    "virtual reality": "virtualreality",
    "augmented reality": "augmentedreality",
    "climate change": "climatechange",
    "energy systems": "energysystems",
    "back propagation": "backpropagation",
    'supply chain': 'supplychain',
    'capstone project': 'capstoneproject',
    'material science': 'materialscience',
    'special topics': 'specialtopics',
    'enviromental engineering':'enviromentalengineering',
    'image processing':'imageprocessing',
    'organic chemistry':'organicchemistry',
    'numerical stability': 'numericalstability',
    'first year': 'firstyear',
    'mccormick firstyear experience':'mccormickfirstyearexperience',
    'natural language processing':'naturallanguageprocessing',
    'neural networks':'neuralnetworks',
    'operating systems':'operatingsystems',
    'computer security': 'computersecurity',
    'personal develop':'personaldevelopment'

}
# Function to handle bigram replacements
def replace_bigrams(tokens, bigram_replacements):
    i = 0
    new_tokens = []
    while i < len(tokens):
        if i < len(tokens) - 1 and ' '.join([tokens[i], tokens[i+1]]) in bigram_replacements:
            new_tokens.append(bigram_replacements[' '.join([tokens[i], tokens[i+1]])])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def preprocess_text(description):
    description = description.lower().replace('/', ' ').replace('-', ' ')
    tokens = word_tokenize(description)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = replace_custom_words(tokens)
    tokens = replace_bigrams(tokens, bigram_replacements)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return ' '.join(filtered_tokens)

from datetime import datetime, timedelta
# Define the mapping of technical feature names to human-readable names
feature_name_mapping = {
    'LSA_1': 'Engineering Analysis',
    'LSA_2': 'Introductory Material',
    'LSA_3': 'Fluid Mechanics',
    'LSA_4': 'Discourse Model',
    'LSA_5': 'System Design',
    'LSA_6': 'Mechanical Design',
    'LSA_7': 'Seminar Structure',
    'LSA_8': 'Design Analysis',
    'LSA_9': 'Statistical Learning',
    'LSA_10': 'Environmental Topics',
    'LSA_11': 'Advanced Analysis',
    'LSA_12': 'Mechanical Selection',
    'LSA_13': 'Research Training',
    'LSA_14': 'Manufacturing Process',
    'LSA_15': 'Optimization Modeling',
    'LSA_16': 'Thermodynamics Materials',
    'LSA_17': 'Probability Thermodynamics',
    'LSA_18': 'Project Management',
    'LSA_19': 'Advanced Manufacturing',
    'LSA_20': 'Transportation Fundamentals',
    'LSA_21': 'Entrepreneurial Principles',
    'LSA_22': 'Mechanical Simulation',
    'LSA_23': 'Modeling Simulation',
    'LSA_24': 'Computing Program',
    'LSA_25': 'Advanced Physics',
    'LSA_26': 'Structural Environmental',
    'LSA_27': 'Biomedical Advances',
    'LSA_28': 'Biomedical Simulation',
    'LSA_29': 'Experimental Events',
    'LSA_30': 'Experimental Lab',
    'LSA_31': 'Material Biomedicine',
    'LSA_32': 'Machine Learning',
    'LSA_33': 'Experimental Methods',
    'LSA_34': 'Computational Methods',
    'LSA_35': 'Computational Learning',
    'LSA_36': 'Micro Dynamics',
    'LSA_37': 'Mathematical Program',
    'LSA_38': 'Mass Transfer',
    'LSA_39': 'Computational Biology',
    'LSA_40': 'Honors Computation',
    'Term': 'Term',
    'Class Type': 'Class Type',
    'Enrollment Capacity': 'Enrollment Capacity',
    'Total Enrollment': 'Total Enrollment',
    'Course Career': 'Course Career',
    'Instructor': 'Instructor',
    'Start Time': 'Start Time',
    'End Time': 'End Time',
    'Mon': 'Monday',
    'Tues': 'Tuesday',
    'Wed': 'Wednesday',
    'Thurs': 'Thursday',
    'Fri': 'Friday',
    'Sat': 'Saturday',
    'Sun': 'Sunday',
    'Subject-Catalog': 'Subject-Catalog',
    'Year': 'Year'
}

# Mapping of LSA components to their respective meaningful names
lsa_feature_names = [
    'Experience_FirstYear_LSA', 'Communication_Design_LSA', 'Topics_PersonalDev_LSA', 
    'Intro_Design_LSA', 'Design_Project_LSA', 'EngAnalysis_LSA', 'DTC_Method_LSA', 
    'Computing_Program_LSA', 'Seminar_Manage_LSA', 'Statistics_Data_LSA', 
    'Material_Process_LSA', 'Systems_Physics_LSA', 'Mechanics_Fluids_LSA', 
    'Project_Management_LSA', 'Selection_MechEng_LSA', 'Selection_Optimization_LSA', 
    'Optimization_Modeling_LSA', 'Analysis_Program_LSA', 'Probabilities_Program_LSA', 
    'Analysis_Human_LSA', 'Data_Structure_LSA', 'Mechanical_Manufacturing_LSA', 
    'Thermodynamic_Env_LSA', 'Environmental_Modeling_LSA', 'Principles_Entrepreneurial_LSA', 
    'Discourse_Statistics_LSA', 'Machine_Learning_LSA', 'Modeling_Stochastic_LSA', 
    'Fundamentals_Transport_LSA', 'Construction_Manage_LSA', 'Develop_Career_LSA', 
    'Advance_Communication_LSA', 'Life_Designing_LSA', 'Life_DesigningComm_LSA', 
    'Structure_Theoretical_LSA', 'ResponsibleConduct_Train_LSA', 'Construction_Computing_LSA', 
    'Advanced_Analytical_LSA', 'Management_Transport_LSA', 'Simulation_Disc_LSA'
]

# Load the TF-IDF vectorizer and SVD model
tfidf_vectorizer = joblib.load(r'C:\Users\marko\Downloads\tfidf_vectorizer.pkl')
svd_model_optimal = joblib.load(r'C:\Users\marko\Downloads\svd_model_optimal.pkl')

app = Flask(__name__)
CORS(app)

def preprocess_features(data):
    # Transform text to TF-IDF
    tfidf_features = tfidf_vectorizer.transform([data['courseDescription']])
    # Transform TF-IDF to LSA features
    lsa_features = svd_model_optimal.transform(tfidf_features).flatten()
    
    features = {
        'Term': data['term'],
        'Subject': data['subject'],
        'Component': data['component'],
        'Instructor': data['instructor'],
        'Start Time': data['startTime'],
        'Mon': 'Mon' in data['daysOfTheWeek'],
        'Tues': 'Tue' in data['daysOfTheWeek'],
        'Wed': 'Wed' in data['daysOfTheWeek'],
        'Thurs': 'Thu' in data['daysOfTheWeek'],
        'Fri': 'Fri' in data['daysOfTheWeek'],
        'Sat': 'Sat' in data['daysOfTheWeek'],
        'Sun': 'Sun' in data['daysOfTheWeek'],
        'Subject-Catalog': f"{data['subject']}_{data['catalog']}",
        'Duration': data['duration'],
        'Class Size Category': data['enrollmentProjections']
    }
    # Append LSA features to the dictionary using descriptive names
    features.update({lsa_feature_names[i]: lsa_features[i] for i in range(len(lsa_features))})
    return features

def predict_for_times(model_input, model, mapping_dict):
    start_times = ['07:00:00', '07:30:00', '08:00:00', '08:30:00', 
                                        '09:00:00', '09:30:00', '10:00:00', '10:30:00',
                                        '11:00:00', '11:30:00', '12:00:00', '12:30:00',  
                                        '13:00:00', '13:30:00', '14:00:00', '14:30:00',  
                                        '15:00:00', '15:30:00',  
                                        '16:00:00', '16:30:00', '17:00:00', '17:30:00',  
                                        '18:00:00', '18:30:00']
    
    time_predictions = {}
    for time in start_times:
        model_input['Start Time'] = time
        for col in model_input.select_dtypes(include=['object']).columns: 
            model_input[col] = model_input[col].map(mapping_dict[col])
        prediction = model.predict(model_input)[0].copy()
        time_predictions[time] = int(round(prediction, 0))
    return time_predictions

def select_model_based_on_range(class_size_category):
    print(class_size_category)
    if class_size_category == '0-30 Students':
        return small_model
    elif class_size_category == '31-70 Students':
        return medium_model
    else:
        return large_model
    
def convert_to_float(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_float(item) for item in obj]
    return obj



    # Load the mapping dictionary from the training phase with 


@app.route('/', methods=['POST'])
def submit_data():
    data = request.json
    features = preprocess_features(data)
    df = pd.DataFrame([features])
    df.drop(df.columns[14], axis=1, inplace=True)
    
    # Select the model based on class size category
    model = select_model_based_on_range(features['Class Size Category'])

    # Load the mapping dictionary from the training phase with 
    mapping_dict = joblib.load(r'C:\Users\marko\Downloads\mapping_dicts.pickle')
    
    # Factorize categorical variables using the loaded mapping 
    for col in df.select_dtypes(include=['object']).columns: 
        df[col] = df[col].map(mapping_dict[col])

    prediction = model.predict(df)
    time_predictions = predict_for_times(df, model, mapping_dict)
    sorted_times = dict(sorted(time_predictions.items(), key=lambda item: item[1], reverse=True))

    # Top 3 times
    top_3_times = dict(list(sorted_times.items())[:3])

    response_data = {
        'forecasted_enrollment': int(round(prediction[0], 0)),
        'chart_data': [convert_to_float(sorted_times)],
        'times': [convert_to_float(top_3_times)]
    }
    
    return jsonify(response_data)



if __name__ == '__main__':
    app.run(debug=True)

