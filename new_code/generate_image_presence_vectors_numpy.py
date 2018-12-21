
#Using presence of images as features
train_vectors = {}
test_vectors = {}

train_vectors_numerized = []
test_vectors_numerized = []

for article in root_train_file.getchildren():
	if re.search(r'images*',' '.join(e for e in article.itertext()).lower().encode('utf-8')):
		train_vectors[article.attrib['id']] = 1
	else:
		train_vectors[article.attrib['id']] = 0

for article in root_test_file.getchildren():
	if re.search(r'images*',' '.join(e for e in article.itertext()).lower().encode('utf-8')):
		test_vectors[article.attrib['id']] = 1
	else:
		test_vectors[article.attrib['id']] = 0

for i in sorted(train_vectors.items(), key=lambda x:x[0]):
	train_vectors_numerized.append([i[1]])

for i in sorted(test_vectors.items(), key=lambda x:x[0]):
	test_vectors_numerized.append([i[1]])