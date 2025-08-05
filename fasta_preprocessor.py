import sys

nome_arquivo = 'sequences_translated.fa' if len(sys.argv) == 3 else 'Noncoding.fa'
print ("Tamanho do argumento: len(sys.argv) =  ",len(sys.argv))
strings = list()
string = ''
with open(nome_arquivo) as f:
    for line in f:
        line = line.rstrip()
        if len(line) == 0:
            continue
        if line[0] == '>':
            if string != '':
                strings.append(string)
            string = line + '#####'
        else:
            string += line
strings.append(string)

with open(nome_arquivo + '.simplified', 'w') as f:
    for s in strings:
        f.write(s + '\n')
