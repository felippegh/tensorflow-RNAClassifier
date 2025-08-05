from itertools import product
import sys, csv
quantidade = 5
nome_arquivo = 'Noncoding.fa' if len(sys.argv) == 1 else 'sequences_translated.fa'
eh_codificante = 0 if len(sys.argv) == 1 else 1
conjuntos = list()
letras = 'ACGT'

for i in range(1, quantidade):
    conjuntos.extend(map(''.join, product(letras, repeat=i)))

sequencias = list()
chave_temp = ''
genes_temp = ''
with open(nome_arquivo) as f:
    for line in f:
        line = line.rstrip().upper()
        if len(line) == 0:
            continue
        if line[0] == '>':
            if chave_temp != '':
                sequencias.append({chave_temp: genes_temp})
            chave_temp = line
            genes_temp = ''
        else:
            genes_temp += line
sequencias.append({chave_temp: genes_temp})

def doSomething(string):
    aux = dict()
    for c in conjuntos:
        aux[c] = 0
    lero = {
        'R': 'A',
        'Y': 'C',
        'S': 'G',
        'W': 'A',
        'K': 'G',
        'M': 'A',
        'B': 'C',
        'D': 'A',
        'H': 'A',
        'V': 'A',
        'N': 'A',
        'U': 'T',
        '.': 'A',
        '-': 'A'
    }
    for k, v in lero.items():
        string = string.replace(k, v)
    caracteres_invalidos = 'EFIJLOPQXZ1234567890'
    for c_i in caracteres_invalidos:
        if c_i in string:
            return aux
    for i in range(0, quantidade - 1):
        for j in range(i, len(string)):
            index = string[j - i: j + 1]
            aux[index] += 1
    for c in conjuntos:
        aux[c] = aux[c] / len(string)
    return aux

meu_csv = list()
for sequencia in sequencias:
    for k, v in sequencia.items():
        aux = doSomething(v)
        meu_csv.append({k: aux})

with open('test_' + nome_arquivo + '.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    temp = ['ID Proteína', 'Codificante']
    temp.extend(conjuntos)
    data = [temp]

    for aux in meu_csv:
        for k, v in aux.items():
            temp = [k, eh_codificante]
            for c in conjuntos:
                temp.append(v[c])
            data.append(temp)

    writer.writerows(data)
