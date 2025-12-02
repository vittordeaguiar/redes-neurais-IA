#!/usr/bin/env python3
"""
Gerador de Dataset com Ruído para Reconhecimento de Letras
Gera variações com ruído das 26 letras do alfabeto em matriz 5x5
"""

import random
from typing import List, Tuple

# Letras base (padrão limpo)
LETRAS_BASE = {
    'A': [-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1],
    'B': [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1],
    'C': [-1,1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1],
    'D': [1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1],
    'E': [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1],
    'F': [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
    'G': [-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1],
    'H': [1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1],
    'I': [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1],
    'J': [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1],
    'K': [1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1],
    'L': [1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1],
    'M': [1,-1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1],
    'N': [1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1],
    'O': [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1],
    'P': [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1],
    'Q': [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,1],
    'R': [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1],
    'S': [-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1],
    'T': [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],
    'U': [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1],
    'V': [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1],
    'W': [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1],
    'X': [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1],
    'Y': [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1],
    'Z': [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1],
}


def adicionar_ruido(padrao: List[int], num_pixels: int = 1) -> List[int]:
    """
    Adiciona ruído invertendo aleatoriamente alguns pixels.
    
    Args:
        padrao: Lista de 25 valores (-1 ou 1)
        num_pixels: Quantidade de pixels a inverter
    
    Returns:
        Nova lista com ruído aplicado
    """
    novo_padrao = padrao.copy()
    indices = random.sample(range(25), num_pixels)
    for i in indices:
        novo_padrao[i] = -novo_padrao[i]  # Inverte: 1 -> -1 ou -1 -> 1
    return novo_padrao


def gerar_variacoes(letra: str, padrao: List[int], num_variacoes: int = 5) -> List[Tuple[List[int], str]]:
    """
    Gera variações com ruído para uma letra.
    
    Args:
        letra: A letra (A-Z)
        padrao: Padrão base da letra
        num_variacoes: Quantas variações criar
    
    Returns:
        Lista de tuplas (padrao_com_ruido, letra)
    """
    variacoes = []
    
    # Primeira variação: padrão original (sem ruído)
    variacoes.append((padrao.copy(), letra))
    
    # Variações com diferentes níveis de ruído
    niveis_ruido = [1, 1, 2, 2, 3]  # 1-3 pixels de ruído
    
    for i in range(num_variacoes - 1):
        nivel = niveis_ruido[i % len(niveis_ruido)]
        padrao_ruido = adicionar_ruido(padrao, nivel)
        variacoes.append((padrao_ruido, letra))
    
    return variacoes


def gerar_arff(filename: str, variacoes_por_letra: int = 6):
    """Gera arquivo ARFF completo com todas as letras e variações."""
    
    with open(filename, 'w') as f:
        # Header ARFF
        f.write("@relation letras_alfabeto_com_ruido\n\n")
        
        for i in range(1, 26):
            f.write(f"@attribute n{i} {{-1, 1}}\n")
        
        f.write("@attribute classe {A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}\n\n")
        f.write("@data\n")
        
        # Gerar dados
        total_registros = 0
        for letra, padrao in LETRAS_BASE.items():
            variacoes = gerar_variacoes(letra, padrao, variacoes_por_letra)
            for padrao_var, _ in variacoes:
                linha = ','.join(str(v) for v in padrao_var) + f",{letra}\n"
                f.write(linha)
                total_registros += 1
        
        print(f"Arquivo {filename} gerado com {total_registros} registros!")


def gerar_fatos_prolog(filename: str, variacoes_por_letra: int = 6):
    """Gera arquivo com fatos Prolog."""
    
    with open(filename, 'w') as f:
        f.write("% Dataset de letras para rede neural\n")
        f.write("% Formato: letra(Classe, [N1,N2,...,N25]).\n\n")
        
        total_registros = 0
        for letra, padrao in LETRAS_BASE.items():
            variacoes = gerar_variacoes(letra, padrao, variacoes_por_letra)
            f.write(f"% Letra {letra}\n")
            for padrao_var, _ in variacoes:
                valores = ','.join(str(v) for v in padrao_var)
                f.write(f"letra({letra.lower()}, [{valores}]).\n")
                total_registros += 1
            f.write("\n")
        
        print(f"Arquivo {filename} gerado com {total_registros} fatos!")


def visualizar_letra(padrao: List[int], titulo: str = ""):
    """Exibe visualmente uma letra no console."""
    if titulo:
        print(f"\n{titulo}")
    print("-" * 11)
    for i in range(5):
        linha = ""
        for j in range(5):
            idx = i * 5 + j
            linha += "█ " if padrao[idx] == 1 else ". "
        print(f"| {linha}|")
    print("-" * 11)


def demo_ruido():
    """Demonstra o efeito do ruído em uma letra."""
    print("\n" + "="*50)
    print("DEMONSTRAÇÃO DE RUÍDO NA LETRA 'A'")
    print("="*50)
    
    padrao_a = LETRAS_BASE['A']
    
    visualizar_letra(padrao_a, "Letra A - Original")
    
    for nivel in [1, 2, 3]:
        padrao_ruido = adicionar_ruido(padrao_a, nivel)
        visualizar_letra(padrao_ruido, f"Letra A - Com {nivel} pixel(s) de ruído")


if __name__ == "__main__":
    import sys
    
    # Seed para reprodutibilidade (opcional - comente para aleatoriedade total)
    random.seed(42)
    
    print("="*60)
    print("GERADOR DE DATASET - LETRAS DO ALFABETO 5x5")
    print("="*60)
    
    # Parâmetros
    variacoes = 6  # 6 variações por letra = 156 registros total
    
    # Gerar arquivos
    gerar_arff("letras_dataset.arff", variacoes)
    gerar_fatos_prolog("letras_dataset.pl", variacoes)
    
    # Demo visual
    demo_ruido()
    
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    print(f"Total de letras: 26")
    print(f"Variações por letra: {variacoes}")
    print(f"Total de registros: {26 * variacoes}")
    print(f"\nArquivos gerados:")
    print("  - letras_dataset.arff (formato WEKA)")
    print("  - letras_dataset.pl (fatos Prolog)")
