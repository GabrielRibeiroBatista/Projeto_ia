#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de predição para o projeto de IA.

Este módulo contém funções para carregar um modelo treinado e realizar
inferências em novos dados.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple


def carregar_modelo(caminho_modelo: str) -> Dict[str, Any]:
    """
    Carrega um modelo treinado a partir de um arquivo.

    Args:
        caminho_modelo: Caminho para o arquivo do modelo.

    Returns:
        Dicionário contendo o modelo e seus metadados.
        
    Raises:
        FileNotFoundError: Se o arquivo do modelo não for encontrado.
    """
    print(f"Carregando modelo de: {caminho_modelo}")
    
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {caminho_modelo}")
    
    modelo_info = joblib.load(caminho_modelo)
    
    print(f"Modelo carregado com sucesso: {modelo_info['modelo'].__class__.__name__}")
    print(f"Data de criação: {modelo_info.get('data_criacao', 'Não disponível')}")
    
    if 'metadata' in modelo_info and modelo_info['metadata']:
        print("\nMetadados do modelo:")
        for chave, valor in modelo_info['metadata'].items():
            if isinstance(valor, (dict, list, tuple, np.ndarray)):
                print(f"  {chave}: {type(valor)}")
            else:
                print(f"  {chave}: {valor}")
    
    return modelo_info


def preparar_dados_predicao(
    dados: Union[pd.DataFrame, np.ndarray, List[List[float]]],
    scaler: Any,
    feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    Prepara novos dados para predição, aplicando as mesmas transformações
    usadas durante o treinamento.

    Args:
        dados: Dados de entrada para predição.
        scaler: Objeto de normalização usado nos dados de treino.
        feature_names: Lista com os nomes das features (opcional).

    Returns:
        Array NumPy com os dados preparados para predição.
    """
    print("Preparando dados para predição...")
    
    # Converter para DataFrame se for array ou lista
    if isinstance(dados, (np.ndarray, list)):
        if feature_names and len(feature_names) == (dados.shape[1] if isinstance(dados, np.ndarray) else len(dados[0])):
            dados = pd.DataFrame(dados, columns=feature_names)
        else:
            dados = pd.DataFrame(dados)
    
    # Verificar se todas as features necessárias estão presentes
    if feature_names and not all(feat in dados.columns for feat in feature_names):
        missing_features = [feat for feat in feature_names if feat not in dados.columns]
        raise ValueError(f"Features ausentes nos dados: {missing_features}")
    
    # Selecionar apenas as features relevantes na ordem correta
    if feature_names:
        dados = dados[feature_names]
    
    # Converter para array NumPy
    X = dados.values
    
    # Aplicar normalização
    X_norm = scaler.transform(X)
    
    print(f"Dados preparados com formato: {X_norm.shape}")
    return X_norm


def fazer_predicao(
    modelo_info: Dict[str, Any],
    dados: Union[pd.DataFrame, np.ndarray, List[List[float]]],
    retornar_probabilidades: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Realiza predições em novos dados usando o modelo carregado.

    Args:
        modelo_info: Dicionário contendo o modelo e seus metadados.
        dados: Dados de entrada para predição.
        retornar_probabilidades: Se True, retorna também as probabilidades das classes.

    Returns:
        Array com as predições ou tupla (predições, probabilidades).
    """
    print("Realizando predições...")
    
    # Extrair componentes do modelo
    modelo = modelo_info['modelo']
    scaler = modelo_info['scaler']
    feature_names = modelo_info.get('feature_names')
    
    # Preparar dados
    X_prep = preparar_dados_predicao(dados, scaler, feature_names)
    
    # Fazer predições
    y_pred = modelo.predict(X_prep)
    
    # Obter probabilidades (se solicitado e disponível)
    if retornar_probabilidades and hasattr(modelo, 'predict_proba'):
        y_proba = modelo.predict_proba(X_prep)
        print(f"Predições concluídas para {len(y_pred)} amostras, com probabilidades.")
        return y_pred, y_proba
    
    print(f"Predições concluídas para {len(y_pred)} amostras.")
    return y_pred


def interpretar_resultados(
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    mapeamento_classes: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Interpreta os resultados da predição em um formato mais amigável.

    Args:
        y_pred: Array com as predições.
        y_proba: Array com as probabilidades das classes (opcional).
        mapeamento_classes: Dicionário mapeando índices de classes para nomes (opcional).

    Returns:
        DataFrame com os resultados interpretados.
    """
    print("Interpretando resultados...")
    
    # Criar dicionário base para resultados
    resultados = {'predicao': y_pred}
    
    # Adicionar nomes das classes se fornecido
    if mapeamento_classes:
        resultados['classe'] = [mapeamento_classes.get(int(pred), f"Classe {pred}") for pred in y_pred]
    
    # Adicionar probabilidades se disponíveis
    if y_proba is not None:
        for i in range(y_proba.shape[1]):
            classe_nome = mapeamento_classes.get(i, f"Classe {i}") if mapeamento_classes else f"Classe {i}"
            resultados[f'prob_{classe_nome}'] = y_proba[:, i]
    
    # Criar DataFrame
    df_resultados = pd.DataFrame(resultados)
    
    print("Interpretação concluída.")
    return df_resultados


def prever_novos_dados(
    caminho_modelo: str,
    dados: Union[pd.DataFrame, np.ndarray, List[List[float]]],
    mapeamento_classes: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Função principal que orquestra todo o processo de predição.

    Args:
        caminho_modelo: Caminho para o arquivo do modelo.
        dados: Dados de entrada para predição.
        mapeamento_classes: Dicionário mapeando índices de classes para nomes (opcional).

    Returns:
        DataFrame com os resultados das predições.
    """
    # Carregar modelo
    modelo_info = carregar_modelo(caminho_modelo)
    
    # Fazer predições com probabilidades
    y_pred, y_proba = fazer_predicao(modelo_info, dados, retornar_probabilidades=True)
    
    # Interpretar resultados
    resultados = interpretar_resultados(y_pred, y_proba, mapeamento_classes)
    
    return resultados


if __name__ == "__main__":
    # Testar o módulo quando executado diretamente
    from preprocess import gerar_dados_sinteticos
    
    # Gerar alguns dados de exemplo
    df = gerar_dados_sinteticos(n_amostras=10, n_features=10)
    X_exemplo = df.drop(columns=['target']).values[:5]  # Pegar apenas 5 amostras
    
    # Definir mapeamento de classes
    mapeamento = {0: "Reprovado", 1: "Aprovado"}
    
    try:
        # Tentar carregar um modelo existente
        resultados = prever_novos_dados(
            '../models/modelo_teste.pkl',
            X_exemplo,
            mapeamento_classes=mapeamento
        )
        
        print("\nResultados da predição:")
        print(resultados)
        
    except FileNotFoundError:
        print("\nModelo não encontrado. Execute primeiro o módulo train.py para criar um modelo.")
