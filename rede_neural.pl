% ============================================================================
% REDE NEURAL PERCEPTRON PARA RECONHECIMENTO DE LETRAS
% Trabalho de Inteligência Artificial
% ============================================================================
%
% Este código implementa um Perceptron multicamada para classificar
% as 26 letras do alfabeto a partir de representações matriciais 5x5.
%
% Para executar:
%   1. Abra o SWI-Prolog
%   2. Carregue este arquivo: ?- [rede_neural].
%   3. Execute o menu: ?- menu.
%
% ============================================================================

:- dynamic peso/3.          % peso(Classe, Indice, Valor)
:- dynamic bias/2.          % bias(Classe, Valor)
:- dynamic taxa_aprendizado/1.
:- dynamic epoca_atual/1.

% ============================================================================
% CONFIGURAÇÕES INICIAIS
% ============================================================================

% Taxa de aprendizado padrão
taxa_aprendizado(0.1).

% Lista de todas as classes (letras)
classes([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z]).

% ============================================================================
% DATASET - LETRAS BASE (sem ruído)
% ============================================================================

% Formato: exemplo(Classe, [N1,N2,...,N25])

% Letra A
exemplo(a, [-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(a, [-1,1,1,1,-1,1,1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(a, [-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,1,-1,1]).
exemplo(a, [-1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(a, [-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,-1,1,-1,-1,1]).
exemplo(a, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,1]).

% Letra B
exemplo(b, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(b, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(b, [1,1,1,-1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(b, [1,1,-1,1,-1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(b, [-1,1,1,1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(b, [1,1,1,1,-1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,1,1,1,1,1,1,1,-1]).

% Letra C
exemplo(c, [-1,1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1]).
exemplo(c, [-1,1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,1]).
exemplo(c, [-1,1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1]).
exemplo(c, [1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1]).
exemplo(c, [-1,1,1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,1]).
exemplo(c, [-1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1]).

% Letra D
exemplo(d, [1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(d, [1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(d, [1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(d, [1,1,-1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,1,1,-1]).
exemplo(d, [1,1,1,1,-1,1,-1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(d, [1,1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1,1,-1]).

% Letra E
exemplo(e, [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1]).
exemplo(e, [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,1,1]).
exemplo(e, [1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1]).
exemplo(e, [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1]).
exemplo(e, [1,1,1,1,1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1]).
exemplo(e, [1,1,1,1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,1]).

% Letra F
exemplo(f, [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1]).
exemplo(f, [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]).
exemplo(f, [1,1,1,1,1,1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1]).
exemplo(f, [1,1,1,-1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1]).
exemplo(f, [1,1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1]).
exemplo(f, [1,1,1,1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1]).

% Letra G
exemplo(g, [-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).
exemplo(g, [-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,1,1,-1]).
exemplo(g, [-1,1,1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).
exemplo(g, [1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).
exemplo(g, [-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1]).
exemplo(g, [-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).

% Letra H
exemplo(h, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(h, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1]).
exemplo(h, [1,-1,-1,-1,1,1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(h, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1]).
exemplo(h, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(h, [1,-1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).

% Letra I
exemplo(i, [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1]).
exemplo(i, [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,1,1]).
exemplo(i, [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1,1]).
exemplo(i, [1,1,1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1]).
exemplo(i, [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,1]).
exemplo(i, [1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,1,1,1]).

% Letra J
exemplo(j, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1]).
exemplo(j, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1]).
exemplo(j, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,-1]).
exemplo(j, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1]).
exemplo(j, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,-1]).
exemplo(j, [1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1]).

% Letra K
exemplo(k, [1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1]).
exemplo(k, [1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,1]).
exemplo(k, [1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,1,1]).
exemplo(k, [1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1]).
exemplo(k, [1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1]).
exemplo(k, [1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1]).

% Letra L
exemplo(l, [1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1]).
exemplo(l, [1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1]).
exemplo(l, [1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1]).
exemplo(l, [1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1]).
exemplo(l, [1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1]).
exemplo(l, [1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1,1]).

% Letra M
exemplo(m, [1,-1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(m, [1,-1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1]).
exemplo(m, [1,-1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1]).
exemplo(m, [1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(m, [1,-1,-1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1]).
exemplo(m, [1,-1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1]).

% Letra N
exemplo(n, [1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1]).
exemplo(n, [1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,1]).
exemplo(n, [1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1]).
exemplo(n, [1,-1,-1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1]).
exemplo(n, [1,-1,-1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1]).
exemplo(n, [1,-1,-1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1]).

% Letra O
exemplo(o, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).
exemplo(o, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,-1]).
exemplo(o, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1]).
exemplo(o, [1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).
exemplo(o, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,-1,-1]).
exemplo(o, [-1,1,1,1,-1,1,-1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).

% Letra P
exemplo(p, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1]).
exemplo(p, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,-1]).
exemplo(p, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1]).
exemplo(p, [1,1,1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1]).
exemplo(p, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1]).
exemplo(p, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1]).

% Letra Q
exemplo(q, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,1]).
exemplo(q, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,1,1,-1,1]).
exemplo(q, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,1,1]).
exemplo(q, [1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,1]).
exemplo(q, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,1]).
exemplo(q, [-1,1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,1,-1,1]).

% Letra R
exemplo(r, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1]).
exemplo(r, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,1]).
exemplo(r, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,1,1]).
exemplo(r, [1,1,1,1,-1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1]).
exemplo(r, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1]).
exemplo(r, [1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1]).

% Letra S
exemplo(s, [-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(s, [-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,1,1,-1]).
exemplo(s, [-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,-1]).
exemplo(s, [1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1]).
exemplo(s, [-1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1]).
exemplo(s, [-1,1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1]).

% Letra T
exemplo(t, [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]).
exemplo(t, [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1]).
exemplo(t, [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1]).
exemplo(t, [1,1,1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]).
exemplo(t, [1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1]).
exemplo(t, [1,1,-1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]).

% Letra U
exemplo(u, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).
exemplo(u, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,-1]).
exemplo(u, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1]).
exemplo(u, [1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1]).
exemplo(u, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,-1,-1]).
exemplo(u, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,-1]).

% Letra V
exemplo(v, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1]).
exemplo(v, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1]).
exemplo(v, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,1]).
exemplo(v, [1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1]).
exemplo(v, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1]).
exemplo(v, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1]).

% Letra W
exemplo(w, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1]).
exemplo(w, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,1,-1,-1,-1,1]).
exemplo(w, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1]).
exemplo(w, [1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,-1,1]).
exemplo(w, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,1]).
exemplo(w, [1,-1,-1,-1,1,1,-1,-1,-1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,-1,-1,1]).

% Letra X
exemplo(x, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1]).
exemplo(x, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1]).
exemplo(x, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1]).
exemplo(x, [1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1]).
exemplo(x, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1]).
exemplo(x, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1]).

% Letra Y
exemplo(y, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]).
exemplo(y, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1]).
exemplo(y, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,1]).
exemplo(y, [1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]).
exemplo(y, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1]).
exemplo(y, [1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1]).

% Letra Z
exemplo(z, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1]).
exemplo(z, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,1,1]).
exemplo(z, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1]).
exemplo(z, [1,1,1,1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,1,1]).
exemplo(z, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,1]).
exemplo(z, [1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,1,1,1,1,1]).

% ============================================================================
% FUNÇÕES DE ATIVAÇÃO
% ============================================================================

% Função degrau (step function)
% Retorna 1 se X >= 0, senão -1
funcao_ativacao(X, 1) :- X >= 0, !.
funcao_ativacao(_, -1).

% Função sigmoide (alternativa - não usada por padrão)
sigmoid(X, Y) :- Y is 1 / (1 + exp(-X)).

% ============================================================================
% INICIALIZAÇÃO DOS PESOS
% ============================================================================

% Inicializa pesos aleatórios entre -0.5 e 0.5
inicializar_peso(Peso) :-
    random(R),
    Peso is R - 0.5.

% Inicializa todos os pesos para uma classe
inicializar_pesos_classe(Classe) :-
    retractall(peso(Classe, _, _)),
    retractall(bias(Classe, _)),
    inicializar_peso(B),
    assertz(bias(Classe, B)),
    inicializar_pesos_entradas(Classe, 1, 25).

inicializar_pesos_entradas(_, I, Max) :- I > Max, !.
inicializar_pesos_entradas(Classe, I, Max) :-
    inicializar_peso(P),
    assertz(peso(Classe, I, P)),
    I1 is I + 1,
    inicializar_pesos_entradas(Classe, I1, Max).

% Inicializa pesos para todas as classes
inicializar_rede :-
    write('Inicializando rede neural...'), nl,
    retractall(peso(_, _, _)),
    retractall(bias(_, _)),
    classes(Lista),
    inicializar_todas_classes(Lista),
    write('Rede inicializada com sucesso!'), nl.

inicializar_todas_classes([]).
inicializar_todas_classes([Classe|Resto]) :-
    inicializar_pesos_classe(Classe),
    inicializar_todas_classes(Resto).

% ============================================================================
% CÁLCULO DA SAÍDA (FORWARD PASS)
% ============================================================================

% Calcula o produto escalar entre entradas e pesos
produto_escalar(_, [], [], 0) :- !.
produto_escalar(Classe, [E|Entradas], [I|Indices], Soma) :-
    peso(Classe, I, P),
    produto_escalar(Classe, Entradas, Indices, SomaResto),
    Soma is SomaResto + (E * P).

% Gera lista de índices [1,2,3,...,25]
gerar_indices(Indices) :-
    numlist(1, 25, Indices).

% Calcula a saída para uma classe específica
calcular_saida(Classe, Entradas, Saida) :-
    gerar_indices(Indices),
    produto_escalar(Classe, Entradas, Indices, Soma),
    bias(Classe, B),
    SomaTotal is Soma + B,
    funcao_ativacao(SomaTotal, Saida).

% Calcula saída bruta (antes da ativação) - útil para comparação
calcular_saida_bruta(Classe, Entradas, SaidaBruta) :-
    gerar_indices(Indices),
    produto_escalar(Classe, Entradas, Indices, Soma),
    bias(Classe, B),
    SaidaBruta is Soma + B.

% ============================================================================
% CLASSIFICAÇÃO (PREDIÇÃO)
% ============================================================================

% Encontra a classe com maior saída bruta
classificar(Entradas, ClassePredita) :-
    classes(Lista),
    encontrar_melhor_classe(Lista, Entradas, _, ClassePredita).

encontrar_melhor_classe([Classe], Entradas, Saida, Classe) :-
    calcular_saida_bruta(Classe, Entradas, Saida), !.
encontrar_melhor_classe([Classe|Resto], Entradas, MelhorSaida, MelhorClasse) :-
    calcular_saida_bruta(Classe, Entradas, Saida),
    encontrar_melhor_classe(Resto, Entradas, SaidaResto, ClasseResto),
    (Saida > SaidaResto ->
        MelhorSaida = Saida, MelhorClasse = Classe
    ;
        MelhorSaida = SaidaResto, MelhorClasse = ClasseResto
    ).

% ============================================================================
% TREINAMENTO (AJUSTE DE PESOS)
% ============================================================================

% Treina um exemplo para uma classe específica
treinar_exemplo_classe(Classe, Entradas, Esperado) :-
    calcular_saida(Classe, Entradas, Obtido),
    Erro is Esperado - Obtido,
    (Erro \= 0 ->
        ajustar_pesos(Classe, Entradas, Erro)
    ;
        true
    ).

% Ajusta os pesos baseado no erro
ajustar_pesos(Classe, Entradas, Erro) :-
    taxa_aprendizado(Taxa),
    gerar_indices(Indices),
    ajustar_pesos_lista(Classe, Entradas, Indices, Erro, Taxa),
    ajustar_bias(Classe, Erro, Taxa).

ajustar_pesos_lista(_, [], [], _, _) :- !.
ajustar_pesos_lista(Classe, [E|Entradas], [I|Indices], Erro, Taxa) :-
    peso(Classe, I, PesoAtual),
    NovoPeso is PesoAtual + (Taxa * Erro * E),
    retract(peso(Classe, I, PesoAtual)),
    assertz(peso(Classe, I, NovoPeso)),
    ajustar_pesos_lista(Classe, Entradas, Indices, Erro, Taxa).

ajustar_bias(Classe, Erro, Taxa) :-
    bias(Classe, BiasAtual),
    NovoBias is BiasAtual + (Taxa * Erro),
    retract(bias(Classe, BiasAtual)),
    assertz(bias(Classe, NovoBias)).

% Treina um exemplo para todas as classes (one-vs-all)
treinar_exemplo(ClasseReal, Entradas) :-
    classes(Lista),
    treinar_para_todas_classes(Lista, ClasseReal, Entradas).

treinar_para_todas_classes([], _, _) :- !.
treinar_para_todas_classes([Classe|Resto], ClasseReal, Entradas) :-
    (Classe == ClasseReal ->
        Esperado = 1
    ;
        Esperado = -1
    ),
    treinar_exemplo_classe(Classe, Entradas, Esperado),
    treinar_para_todas_classes(Resto, ClasseReal, Entradas).

% Treina uma época (todos os exemplos)
treinar_epoca :-
    findall((C, E), exemplo(C, E), Exemplos),
    treinar_lista_exemplos(Exemplos).

treinar_lista_exemplos([]) :- !.
treinar_lista_exemplos([(Classe, Entradas)|Resto]) :-
    treinar_exemplo(Classe, Entradas),
    treinar_lista_exemplos(Resto).

% Treina múltiplas épocas
treinar(NumEpocas) :-
    write('Iniciando treinamento...'), nl,
    treinar_epocas(1, NumEpocas),
    write('Treinamento concluido!'), nl.

treinar_epocas(Atual, Max) :- Atual > Max, !.
treinar_epocas(Atual, Max) :-
    treinar_epoca,
    (0 is Atual mod 10 ->
        format('Epoca ~w/~w concluida~n', [Atual, Max])
    ;
        true
    ),
    Prox is Atual + 1,
    treinar_epocas(Prox, Max).

% ============================================================================
% AVALIAÇÃO
% ============================================================================

% Testa todos os exemplos e calcula acurácia
avaliar :-
    findall((C, E), exemplo(C, E), Exemplos),
    length(Exemplos, Total),
    contar_acertos(Exemplos, 0, Acertos),
    Acuracia is (Acertos / Total) * 100,
    format('~nResultados da Avaliacao:~n'),
    format('Total de exemplos: ~w~n', [Total]),
    format('Acertos: ~w~n', [Acertos]),
    format('Acuracia: ~2f%~n', [Acuracia]).

contar_acertos([], Acc, Acc) :- !.
contar_acertos([(ClasseReal, Entradas)|Resto], Acc, Total) :-
    classificar(Entradas, ClassePredita),
    (ClasseReal == ClassePredita ->
        NovoAcc is Acc + 1
    ;
        NovoAcc = Acc
    ),
    contar_acertos(Resto, NovoAcc, Total).

% Avaliação detalhada mostrando erros
avaliar_detalhado :-
    findall((C, E), exemplo(C, E), Exemplos),
    write('Avaliacao detalhada:'), nl,
    write('-------------------'), nl,
    avaliar_lista_detalhado(Exemplos, 0, 0, Acertos, Total),
    nl,
    Acuracia is (Acertos / Total) * 100,
    format('Total: ~w | Acertos: ~w | Acuracia: ~2f%~n', [Total, Acertos, Acuracia]).

avaliar_lista_detalhado([], A, T, A, T) :- !.
avaliar_lista_detalhado([(ClasseReal, Entradas)|Resto], AccA, AccT, Acertos, Total) :-
    classificar(Entradas, ClassePredita),
    NovoT is AccT + 1,
    (ClasseReal == ClassePredita ->
        NovoA is AccA + 1
    ;
        NovoA = AccA,
        format('ERRO: Real=~w, Predito=~w~n', [ClasseReal, ClassePredita])
    ),
    avaliar_lista_detalhado(Resto, NovoA, NovoT, Acertos, Total).

% ============================================================================
% VISUALIZAÇÃO
% ============================================================================

% Visualiza uma entrada como matriz 5x5
visualizar_matriz(Entradas) :-
    visualizar_linha(Entradas, 1).

visualizar_linha(_, 26) :- !.
visualizar_linha(Entradas, Pos) :-
    nth1(Pos, Entradas, Val),
    (Val == 1 -> write('# ') ; write('. ')),
    (0 is Pos mod 5 -> nl ; true),
    Pos1 is Pos + 1,
    visualizar_linha(Entradas, Pos1).

% Testa uma letra visualmente
testar_visual(Classe) :-
    exemplo(Classe, Entradas),
    format('~nLetra ~w:~n', [Classe]),
    visualizar_matriz(Entradas),
    classificar(Entradas, Predita),
    format('Classificada como: ~w~n', [Predita]),
    (Classe == Predita -> write('CORRETO!') ; write('INCORRETO!')), nl.

% ============================================================================
% CONFIGURAÇÃO DA TAXA DE APRENDIZADO
% ============================================================================

% Altera a taxa de aprendizado
set_taxa_aprendizado(Nova) :-
    retractall(taxa_aprendizado(_)),
    assertz(taxa_aprendizado(Nova)),
    format('Taxa de aprendizado alterada para: ~w~n', [Nova]).

% ============================================================================
% MENU INTERATIVO
% ============================================================================

menu :-
    nl,
    write('========================================'), nl,
    write('  REDE NEURAL - RECONHECIMENTO DE LETRAS'), nl,
    write('========================================'), nl,
    write('1. Inicializar rede (pesos aleatorios)'), nl,
    write('2. Treinar rede (definir numero de epocas)'), nl,
    write('3. Avaliar rede (acuracia)'), nl,
    write('4. Avaliar detalhado (mostra erros)'), nl,
    write('5. Testar letra especifica'), nl,
    write('6. Classificar entrada manual'), nl,
    write('7. Alterar taxa de aprendizado'), nl,
    write('8. Treinar e avaliar (completo)'), nl,
    write('9. Comparar diferentes taxas de aprendizado'), nl,
    write('0. Sair'), nl,
    write('========================================'), nl,
    write('Escolha uma opcao: '),
    read(Opcao),
    processar_opcao(Opcao).

processar_opcao(0) :- write('Ate logo!'), nl, !.
processar_opcao(1) :-
    inicializar_rede,
    menu.
processar_opcao(2) :-
    write('Numero de epocas: '),
    read(N),
    treinar(N),
    menu.
processar_opcao(3) :-
    avaliar,
    menu.
processar_opcao(4) :-
    avaliar_detalhado,
    menu.
processar_opcao(5) :-
    write('Digite a letra (minuscula, ex: a): '),
    read(Letra),
    testar_visual(Letra),
    menu.
processar_opcao(6) :-
    write('Digite os 25 valores da matriz (lista): '),
    read(Entradas),
    classificar(Entradas, Classe),
    format('Classificado como: ~w~n', [Classe]),
    menu.
processar_opcao(7) :-
    write('Nova taxa de aprendizado (ex: 0.1): '),
    read(Taxa),
    set_taxa_aprendizado(Taxa),
    menu.
processar_opcao(8) :-
    write('Numero de epocas: '),
    read(N),
    inicializar_rede,
    treinar(N),
    avaliar,
    menu.
processar_opcao(9) :-
    comparar_taxas,
    menu.
processar_opcao(_) :-
    write('Opcao invalida!'), nl,
    menu.

% ============================================================================
% COMPARAÇÃO DE TAXAS DE APRENDIZADO
% ============================================================================

% Compara diferentes taxas de aprendizado
comparar_taxas :-
    write('Comparando diferentes taxas de aprendizado...'), nl,
    write('(6 epocas para cada taxa)'), nl, nl,
    Taxas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    comparar_lista_taxas(Taxas).

comparar_lista_taxas([]) :- !.
comparar_lista_taxas([Taxa|Resto]) :-
    set_taxa_aprendizado(Taxa),
    inicializar_rede,
    treinar_silencioso(5),
    findall((C, E), exemplo(C, E), Exemplos),
    length(Exemplos, Total),
    contar_acertos(Exemplos, 0, Acertos),
    Acuracia is (Acertos / Total) * 100,
    format('Taxa ~w: ~2f% de acuracia~n', [Taxa, Acuracia]),
    comparar_lista_taxas(Resto).

% Treina sem mensagens
treinar_silencioso(NumEpocas) :-
    treinar_epocas_silencioso(1, NumEpocas).

treinar_epocas_silencioso(Atual, Max) :- Atual > Max, !.
treinar_epocas_silencioso(Atual, Max) :-
    treinar_epoca,
    Prox is Atual + 1,
    treinar_epocas_silencioso(Prox, Max).

% ============================================================================
% EXECUÇÃO RÁPIDA
% ============================================================================

% Executa tudo automaticamente
executar_tudo :-
    inicializar_rede,
    treinar(100),
    avaliar_detalhado.

% Início automático ao carregar
:- write('========================================'), nl,
   write('Rede Neural para Reconhecimento de Letras'), nl,
   write('========================================'), nl,
   write('Digite "menu." para iniciar o menu interativo'), nl,
   write('Ou "executar_tudo." para treinar e avaliar'), nl.
