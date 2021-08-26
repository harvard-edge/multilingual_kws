#%%

alignments = """ab.tar.gz
ar.tar.gz
as.tar.gz
br.tar.gz
ca.tar.gz
cnh.tar.gz
cs.tar.gz
cv.tar.gz
cy.tar.gz
de.tar.gz
dv.tar.gz
el.tar.gz
en.tar.gz
eo.tar.gz
es.tar.gz
et.tar.gz
eu.tar.gz
fa.tar.gz
fr.tar.gz
fy-NL.tar.gz
ga-IE.tar.gz
gn.tar.gz
ha.tar.gz
ia.tar.gz
id.tar.gz
it.tar.gz
ka.tar.gz
ky.tar.gz
lv.tar.gz
mn.tar.gz
mt.tar.gz
nl.tar.gz
or.tar.gz
pl.tar.gz
pt.tar.gz
rm-sursilv.tar.gz
rm-vallader.tar.gz
ro.tar.gz
ru.tar.gz
rw.tar.gz
sah.tar.gz
sk.tar.gz
sl.tar.gz
sv-SE.tar.gz
ta.tar.gz
tr.tar.gz
tt.tar.gz
uk.tar.gz
vi.tar.gz
zh-CN.tar.gz""".split('\n')
# %%

alignments = [i.split('.gz')[0] for i in alignments]

#%%


import subprocess
paths = []

for a in alignments:
	out = subprocess.check_output(['rclone', 'link', '--drive-shared-with-me',f'msc3:MSC/audio/{a}'])
	print(a, out[:-1])
# %%
