#log para verificar a execuxao do scrpit
print("programa iniciado...")


#importando o cliente da API para acessar o serviço spacetrack
from spacetrack import SpaceTrackClient

import pandas as pd
import numpy as np
import requests
import os
import json

from dotenv import load_dotenv

load_dotenv()

IDENTITY = os.getenv("IDENTITY")
PASSWORD = os.getenv("PASSWORD")

CAMINHO_CSV = os.getenv("DATASET_PATH_CSV")

CAMINHO_XLS = os.getenv("DATASET_PATH_XLS")

try:
  #with garante inicio e fim das requisicoes no tempo certo
  with SpaceTrackClient(identity= IDENTITY, password= PASSWORD) as st:
  #requisicao de CDMs públicos com numero de respostas limitado
    cdm_data = st.basicspacedata.cdm_public(
        format = 'json',
        limit = 100)

    #checando o conteudo retornado pela requisicao
    print('dados repondidos pela API: ', cdm_data)

    #transformando a lista de strings em um dict que se comporta como um objeto json
    cdm_data = json.loads(cdm_data)

    #criando df
    df = pd.DataFrame(cdm_data)

    #filtrando conjuncoes entre satelites e detritos
    condicoes_conjuncao = (
        (df['SAT1_OBJECT_TYPE'].isin(['PAYLOAD', 'DEBRIS'])) &
        (df['SAT2_OBJECT_TYPE'].isin(['PAYLOAD', 'DEBRIS'])) &
        (df['SAT1_OBJECT_TYPE'] != df['SAT2_OBJECT_TYPE'])
    )

    df_filtrado = df[condicoes_conjuncao]

    print("df contendo apenas satélites e detritos: ", df_filtrado)

    #extraindo os ids, distancia minima, exclusion volume e tca das conjuncoes e colocando em uma lista de tuplas
    useful_cdm = list(zip(df_filtrado["SAT_1_ID"], df_filtrado["SAT_2_ID"], df_filtrado["MIN_RNG"], df_filtrado["SAT_1_EXCL_VOL"], df_filtrado["SAT_2_EXCL_VOL"], df_filtrado["TCA"]))
    print("lista com os pares de ids e distancia minima de cada conjuncao: ", useful_cdm)
    print("quantidade de tuplas de ID:", len(useful_cdm))

    #definindo um numero limite de itens na lista de CDMs uteis
    MAX_USEFUL_CDM = 10

    #limita o numero de elementos
    while len(useful_cdm) > MAX_USEFUL_CDM:
      useful_cdm.pop()

    print("quantidade de tuplas de ID:", len(useful_cdm))
    print("lista final dos dados de CDM uteis:", useful_cdm)

except Exception as e:
  print('erro ao acessar a API: ', e)

#cria um objeto session para manter a persistência entre requisições
with requests.Session() as session:
  # Credenciais de autenticação para acessar os recursos da API
  login_url = "https://www.space-track.org/ajaxauth/login"
  credentials = {"identity": IDENTITY, "password": PASSWORD}

  response = session.post(login_url, credentials)

  if response.status_code != 200:
      raise Exception("Falha no login! Verifique usuário e senha")
  else:
      print("Login no spacetrack realizado com sucesso")

  # Lista para concatenar responses
  teste_todos_dados_tle = []

  # Customizar query url para receber as variáveis de ID
  for id1, id2, *_ in useful_cdm:
      query_url = f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{id1},{id2}/format/json"
      response = session.get(query_url)

      if response.status_code == 200:
          teste_tle_data = response.json()
          teste_todos_dados_tle.extend(teste_tle_data)
          print("TLEs obtidos com sucesso")

  print("Dados retornados de todas as chamadas da API:", teste_todos_dados_tle)

  # Criando dataframe a partir da lista com todas as responses
  teste_df_tle = pd.DataFrame(teste_todos_dados_tle)

  # Filtrando apenas as colunas relevantes
  teste_df_tle = teste_df_tle[[
      "NORAD_CAT_ID", "OBJECT_TYPE", "OBJECT_NAME",
      "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
      "MEAN_ANOMALY", "SEMIMAJOR_AXIS", "ARG_OF_PERICENTER"
  ]]

  print("Dados de TLE filtrados ")
  print(teste_df_tle)

  #inicializando as listas para receber os dados de CDM
  min_range = []
  sat_1_excl_vol = []
  sat_2_excl_vol = []
  tca = []

  #extraindo cada valor de CDM para lista
  for cdm in useful_cdm:
      min_range.append(cdm[2])
      sat_1_excl_vol.append(cdm[3])
      sat_2_excl_vol.append(cdm[4])
      tca.append(cdm[5])

  #convertendo as listas em series e duplicando os valores de CDM para ficarem correspondentes aos valores de TLE sem repetir o índice
  min_range = pd.Series(min_range).repeat(2).reset_index(drop=True)
  sat_1_excl_vol = pd.Series(sat_1_excl_vol).repeat(2).reset_index(drop=True)
  sat_2_excl_vol = pd.Series(sat_2_excl_vol).repeat(2).reset_index(drop=True)
  tca = pd.Series(tca).repeat(2).reset_index(drop=True)

  #concatenando as series CDM no df de dados TLE
  teste_df_final = pd.concat([teste_df_tle, min_range.rename("MIN_RANGE"), sat_1_excl_vol.rename("SAT_1_EXCL_VOL"), sat_2_excl_vol.rename("SAT_2_EXCL_VOL"), tca.rename("TCA")], axis=1)

  print("Df final com TLE e CDM juntos:")
  print(teste_df_final)

  #Salvando os dados como csv
  teste_df_final.to_csv(CAMINHO_CSV, index=False)
  print("Dataset salvo em csv")

  #Salvando os dados como planilha
  teste_df_final.to_excel(CAMINHO_XLS, index=False)
  print("Dataset salvo em planilha")
  
  
from datetime import datetime, timedelta
from pycaret.regression import setup, compare_models, predict_model

np.set_printoptions(precision=9, suppress=False)

MU_EARTH = 398600.4418  # km^3/s^2


#converte elementos Keplerianos r, v
def kepler_to_rv(a, e, inc, raan, argp, M):

    inc = np.radians(float(inc))
    raan = np.radians(float(raan))
    argp = np.radians(float(argp))
    M = np.radians(float(M))

    E = M
    for _ in range(10):
        E = M + e * np.sin(E)

    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    r = a * (1 - e * np.cos(E))

    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    h = np.sqrt(MU_EARTH * a * (1 - e**2))
    vx_orb = -MU_EARTH / h * np.sin(nu)
    vy_orb = MU_EARTH / h * (e + np.cos(nu))

    R3_W = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan),  np.cos(raan), 0],
        [0, 0, 1]
    ])
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(inc), -np.sin(inc)],
        [0, np.sin(inc),  np.cos(inc)]
    ])
    R3_w = np.array([
        [np.cos(argp), -np.sin(argp), 0],
        [np.sin(argp),  np.cos(argp), 0],
        [0, 0, 1]
    ])

    R = R3_W @ R1_i @ R3_w

    r_vec = R @ np.array([x_orb, y_orb, 0])
    v_vec = R @ np.array([vx_orb, vy_orb, 0])

    return r_vec, v_vec


#calcula a probabilidade de colisão de monte carlo
def monte_carlo_collision_prob(rA, covA, rB, covB, N=50000, collision_radius=10):
    samplesA = np.random.multivariate_normal(rA, covA, N)
    samplesB = np.random.multivariate_normal(rB, covB, N)
    dists = np.linalg.norm(samplesA - samplesB, axis=1)
    return np.mean(dists < collision_radius)


#calcula a probabilidade de colisao para um par de objetos
def compute_pc(paramsA, paramsB, N=30000):

    rA, _ = kepler_to_rv(
        paramsA["SEMIMAJOR_AXIS"],
        paramsA["ECCENTRICITY"],
        paramsA["INCLINATION"],
        paramsA["RA_OF_ASC_NODE"],
        paramsA["ARG_OF_PERICENTER"],
        paramsA["MEAN_ANOMALY"]
    )

    rB, _ = kepler_to_rv(
        paramsB["SEMIMAJOR_AXIS"],
        paramsB["ECCENTRICITY"],
        paramsB["INCLINATION"],
        paramsB["RA_OF_ASC_NODE"],
        paramsB["ARG_OF_PERICENTER"],
        paramsB["MEAN_ANOMALY"]
    )

    sigmaA = np.array([paramsA["ECCENTRICITY"],
                       paramsA["INCLINATION"],
                       paramsA["RA_OF_ASC_NODE"]]) * 0.001

    sigmaB = np.array([paramsB["ECCENTRICITY"],
                       paramsB["INCLINATION"],
                       paramsB["RA_OF_ASC_NODE"]]) * 0.001

    covA = np.diag(sigmaA ** 2)
    covB = np.diag(sigmaB ** 2)

    raioA = paramsA["SAT_1_EXCL_VOL"] ** (1/3)
    raioB = paramsB["SAT_2_EXCL_VOL"] ** (1/3)
    collision_radius = raioA + raioB

    Pc = monte_carlo_collision_prob(
        rA, covA, rB, covB, N=N, collision_radius=collision_radius
    )

    return Pc

#processa todos os pares de dados e transforma em dataframe
def process_dataset(df):
    pc_values = []

    for i in range(0, len(df), 2):
        paramsA = df.iloc[i].to_dict()
        paramsB = df.iloc[i+1].to_dict()

        Pc = compute_pc(paramsA, paramsB)

        pc_values.append(Pc)
        pc_values.append(Pc)

    df["Pc"] = pc_values
    return df


#utiliza o pycaret para realizar previsoes futuras (7 dias)
def run_pycaret_and_forecast(df):
    try:
        #checagem inicial
        if df.empty or "Pc" not in df.columns:
            raise ValueError("DataFrame vazio ou sem coluna 'Pc'")
        
        print(f"Shape original: {df.shape}")
        print(f"NaN no Pc original: {df['Pc'].isnull().sum()}")
        print(f"Valores únicos em Pc: {df['Pc'].unique()}")
        
        #preparação de dados
        df_model = df.copy()

        #removendo TCA dos dados de treinamento
        if "TCA" in df_model.columns:
            df_model["TCA_ts"] = df_model["TCA"].astype("int64") // 10**9
            df_model = df_model.drop(columns=["TCA"])

        #removendo colunas desnecessárias para analise da biblioteca
        cols_to_drop = ['OBJECT_TYPE', 'OBJECT_NAME']
        df_model = df_model.drop(columns=[col for col in cols_to_drop if col in df_model.columns])

        #processando NaN 
        for col in df_model.columns:
            if df_model[col].dtype in ['float64', 'int64']:
                df_model[col] = df_model[col].fillna(df_model[col].median())
            else:
                df_model[col] = df_model[col].fillna(method="ffill").fillna(method="bfill")

        if df_model["Pc"].isnull().any():
            df_model = df_model.dropna(subset=["Pc"])
        
        if df_model.empty:
            raise ValueError("DataFrame ficou vazio após limpeza")

        #verificando se há dados suficientes e variância
        if len(df_model) < 10:
            raise ValueError(f"Dados insuficientes para treinamento: {len[df_model]} linhas")

        #em casos de pouca variação de resultado o processamento não é interrrompido
        if df_model["Pc"].nunique() <= 1:
            print("Coluna Pc tem pouca variação (ou é constante). ")

            
        df_model = df_model.reset_index(drop=True)
        
        print(f"Shape após limpeza: {df_model.shape}")
        print(f"Colunas para modelo: {df_model.columns.tolist()}")

        #setup pycaret
        s = setup(
            data=df_model,
            target="Pc",
            verbose=False,
            session_id=42,
            index=False,
            normalize=True,
            transformation=False,
            remove_multicollinearity=False,
            feature_selection=False,
            polynomial_features=False,
            data_split_shuffle=False,
        )

        #busca o melhor modelo
        from pycaret.regression import compare_models

        print("\n Buscando melhor modelo...")
        best_model = compare_models(sort="RMSE")

        print(f"\n Melhor modelo encontrado: {best_model}")

        #aplicando previsao futura
        max_tca = df["TCA"].max()
        future_dates = [max_tca + timedelta(days=i) for i in range(1, 8)]
        
        last_row = df_model.iloc[-1].copy()
        future_df = pd.DataFrame([last_row] * 7)

        future_df["TCA_ts"] = [d.value // 10**9 for d in future_dates]

        for col in future_df.columns:
            if future_df[col].isnull().any() and col != "Pc":
                if future_df[col].dtype in ['float64', 'int64']:
                    future_df[col] = future_df[col].fillna(future_df[col].median())
                else:
                    future_df[col] = future_df[col].fillna(method="ffill").fillna(method="bfill")

        future_df = future_df.reset_index(drop=True)

        #features usdadas no treinamento
        try:
            feature_names = get_config('X_train').columns.tolist()
        except:
            feature_names = [col for col in df_model.columns if col != "Pc"]
        
        print(f"Features usadas: {feature_names}")

        missing_cols = [c for c in feature_names if c not in future_df.columns]
        for c in missing_cols:
            future_df[c] = 0

        future_df = future_df[feature_names]

        #previsao final
        preds = predict_model(best_model, data=future_df, verbose=False)

        if "prediction_label" not in preds.columns:
            raise ValueError("Coluna 'prediction_label' não encontrada nas previsões")

        future_df["Pc_pred"] = preds["prediction_label"]

        if future_df["Pc_pred"].isnull().sum() > 0:
            fill_value = df_model["Pc"].mean()
            future_df["Pc_pred"] = future_df["Pc_pred"].fillna(fill_value)
        
        print("Previsões realizadas com sucesso!")
        print(f"Valores previstos: {future_df['Pc_pred'].values}")

        return best_model, future_df
        
    except Exception as e:
        print(f"Erro em run_pycaret_and_forecast: {e}")
        import traceback
        traceback.print_exc()
        return None, None


#execucao do algoritmo
df_pred = pd.read_csv(CAMINHO_CSV)


cols_float = [
    "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "MEAN_ANOMALY",
    "SEMIMAJOR_AXIS", "ARG_OF_PERICENTER", "MIN_RANGE",
    "SAT_1_EXCL_VOL", "SAT_2_EXCL_VOL"
]

for col in cols_float:
    df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")

df_pred["TCA"] = pd.to_datetime(df_pred["TCA"], errors="coerce")


resultado = process_dataset(df_pred)

#previsao de 7 dias no futuro
best_model, future_df = run_pycaret_and_forecast(resultado)

#concatenando a series de probabilidade no df final
future_pc_pred = future_df[["Pc_pred"]].copy()

resultado_final = pd.concat([resultado, future_pc_pred], axis=1)

print("dados previstos")
print(future_df)

print("dados finais")
print(resultado)

print("resultado final")
print(resultado_final)