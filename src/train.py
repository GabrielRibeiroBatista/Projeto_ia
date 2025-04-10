#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo de treinamento para o projeto de IA.

Este módulo contém funções para criar, treinar e avaliar modelos de classificação,
além de salvar o modelo treinado para uso posterior.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def criar_modelo(
    tipo_modelo: str = 'random_forest',
    random_state: int = 42,
    **kwargs
) -> Union[RandomForestClassifier, LogisticRegression]:
    """
    Cria um modelo de classificação do tipo especificado.

    Args:
        tipo_modelo: Tipo de modelo a ser criado ('random_forest' ou 'logistic_regression').
        random_state: Semente para reprodutibilidade.
        **kwargs: Parâmetros adicionais específicos para cada tipo de modelo.

    Returns:
        Modelo de classificação instanciado.
        
    Raises:
        ValueError: Se o tipo de modelo não for suportado.
    """
    print(f"Criando modelo de classificação: {tipo_modelo}")
    
    if tipo_modelo == 'random_forest':
        modelo = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            random_state=random_state
        )
    elif tipo_modelo == 'logistic_regression':
        modelo = LogisticRegression(
            C=kwargs.get('C', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            solver=kwargs.get('solver', 'lbfgs'),
            random_state=random_state
        )
    else:
        raise ValueError(f"Tipo de modelo não suportado: {tipo_modelo}")
    
    return modelo


def treinar_modelo(
    modelo: Union[RandomForestClassifier, LogisticRegression],
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Union[RandomForestClassifier, LogisticRegression]:
    """
    Treina o modelo com os dados fornecidos.

    Args:
        modelo: Modelo de classificação a ser treinado.
        X_train: Dados de treino.
        y_train: Rótulos de treino.

    Returns:
        Modelo treinado.
    """
    print(f"Treinando modelo {modelo.__class__.__name__}...")
    
    inicio = time.time()
    modelo.fit(X_train, y_train)
    fim = time.time()
    
    tempo_treino = fim - inicio
    print(f"Treinamento concluído em {tempo_treino:.2f} segundos.")
    
    return modelo


def avaliar_modelo(
    modelo: Union[RandomForestClassifier, LogisticRegression],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Avalia o desempenho do modelo nos dados de teste.

    Args:
        modelo: Modelo treinado a ser avaliado.
        X_test: Dados de teste.
        y_test: Rótulos de teste.
        feature_names: Lista com os nomes das features (opcional).

    Returns:
        Dicionário contendo métricas de avaliação.
    """
    print("Avaliando modelo nos dados de teste...")
    
    # Fazer previsões
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred, output_dict=True)
    matriz_confusao = confusion_matrix(y_test, y_pred)
    
    print(f"Acurácia: {acuracia:.4f}")
    print("Relatório de classificação:")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusão:")
    print(matriz_confusao)
    
    # Extrair importância das features (se disponível)
    importancia_features = {}
    if hasattr(modelo, 'feature_importances_') and feature_names:
        importancia = modelo.feature_importances_
        importancia_features = dict(zip(feature_names, importancia))
        
        # Mostrar as 5 features mais importantes
        print("\nImportância das features (top 5):")
        for feature, imp in sorted(importancia_features.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"{feature}: {imp:.4f}")
    
    # Retornar resultados
    resultados = {
        'acuracia': acuracia,
        'relatorio': relatorio,
        'matriz_confusao': matriz_confusao,
        'importancia_features': importancia_features
    }
    
    return resultados


def salvar_modelo(
    modelo: Union[RandomForestClassifier, LogisticRegression],
    scaler: Any,
    caminho_modelo: str,
    feature_names: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Salva o modelo treinado e metadados associados.

    Args:
        modelo: Modelo treinado a ser salvo.
        scaler: Objeto de normalização usado nos dados.
        caminho_modelo: Caminho onde o modelo será salvo.
        feature_names: Lista com os nomes das features (opcional).
        metadata: Dicionário com metadados adicionais (opcional).

    Returns:
        Caminho completo onde o modelo foi salvo.
    """
    print(f"Salvando modelo em: {caminho_modelo}")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)
    
    # Preparar objeto para salvar
    modelo_info = {
        'modelo': modelo,
        'scaler': scaler,
        'feature_names': feature_names,
        'metadata': metadata or {},
        'data_criacao': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Salvar modelo
    joblib.dump(modelo_info, caminho_modelo)
    print(f"Modelo salvo com sucesso em: {caminho_modelo}")
    
    return caminho_modelo


def treinar_e_avaliar(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: Any,
    feature_names: List[str],
    tipo_modelo: str = 'random_forest',
    caminho_modelo: str = '../models/modelo_treinado.pkl',
    **kwargs
) -> Dict[str, Any]:
    """
    Função principal que orquestra todo o processo de treinamento e avaliação.

    Args:
        X_train: Dados de treino normalizados.
        X_test: Dados de teste normalizados.
        y_train: Rótulos de treino.
        y_test: Rótulos de teste.
        scaler: Objeto de normalização usado nos dados.
        feature_names: Lista com os nomes das features.
        tipo_modelo: Tipo de modelo a ser criado.
        caminho_modelo: Caminho onde o modelo será salvo.
        **kwargs: Parâmetros adicionais para o modelo.

    Returns:
        Dicionário contendo resultados da avaliação e caminho do modelo salvo.
    """
    # Criar modelo
    modelo = criar_modelo(tipo_modelo=tipo_modelo, **kwargs)
    
    # Treinar modelo
    modelo = treinar_modelo(modelo, X_train, y_train)
    
    # Avaliar modelo
    resultados = avaliar_modelo(modelo, X_test, y_test, feature_names)
    
    # Salvar modelo
    caminho_completo = salvar_modelo(
        modelo,
        scaler,
        caminho_modelo,
        feature_names,
        metadata={
            'tipo_modelo': tipo_modelo,
            'parametros': kwargs,
            'acuracia': resultados['acuracia'],
            'tamanho_treino': X_train.shape,
            'tamanho_teste': X_test.shape
        }
    )
    
    # Adicionar caminho do modelo aos resultados
    resultados['caminho_modelo'] = caminho_completo
    
    return resultados


if __name__ == "__main__":
    # Testar o módulo quando executado diretamente
    from preprocess import preparar_dados
    
    # Preparar dados
    X_train, X_test, y_train, y_test, scaler, feature_names = preparar_dados(
        usar_dados_sinteticos=True,
        n_amostras=1000
    )
    
    # Treinar e avaliar modelo
    resultados = treinar_e_avaliar(
        X_train, X_test, y_train, y_test, scaler, feature_names,
        tipo_modelo='random_forest',
        caminho_modelo='../models/modelo_teste.pkl',
        n_estimators=100,
        max_depth=10
    )
    
    print(f"\nModelo salvo em: {resultados['caminho_modelo']}")
    print(f"Acurácia final: {resultados['acuracia']:.4f}")
