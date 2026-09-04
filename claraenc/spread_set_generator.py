bitcount = 12
groups = 4
bits = np.arange(12)
group_bit = bitcount//groups

g_bit_exp = np.arange(groupbit)
#[[g,g,g,g] for g in g_bit_exp]
s_bit_exp = [0,1,-1]
group_asc = np.arange(groups)
first = [(group_asc * s + g) % groups
     for s in s_bit_exp for g in group_asc]

print(np.array(first))