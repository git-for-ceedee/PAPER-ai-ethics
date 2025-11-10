import sqlite3
con = sqlite3.connect('ragEthics.db'); cur = con.cursor()
print('themes:', cur.execute('SELECT COUNT(*) FROM theme').fetchone()[0])
print('chunk_theme links:', cur.execute('SELECT COUNT(*) FROM chunk_theme').fetchone()[0])
con.close()
