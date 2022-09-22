from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
import pandas as pd
from stanza.server import CoreNLPClient
from nltk import tokenize


spark = sparknlp.start()

text = """Patagonia's founder is transferring ownership of the company after nearly 50 years into two entities that will help fight the climate crisis.

Patagonia is a private company based in Ventura, California, that sells outdoor apparel and equipment. Yvon Chouinard founded the company in 1973.
Patagonia said in a press release on Wednesday that, effective immediately, Chouinard and his family will transfer their entire ownership stake into two newly created entities. Those entities will ensure that the company's values will continue to be upheld — and that Patagonia's profits are used to combat climate change.
"If we have any hope of a thriving planet 50 years from now, it demands all of us doing all we can with the resources we have," Chouinard said in a statement Wednesday. "Instead of extracting value from nature and transforming it into wealth, we are using the wealth Patagonia creates to protect the source."
The biggest share of the company — or 98% of Patagonia's stock — will now be under the Holdfast Collective. This nonprofit will make sure that the company's annual profits, about $100 million per year, will be used to "protect nature and biodiversity, support thriving communities and fight the environmental crisis."
The rest of the company's stock will fund the newly created Patagonia Purpose Trust.
This trust will create a permanent legal structure so that the company can never deviate from Chouinard's vision: That a for-profit business can work for the planet.
"Two years ago, the Chouinard family challenged a few of us to develop a new structure with two central goals," said Patagonia CEO Ryan Gellert in the press release. "They wanted us to both protect the purpose of the business and immediately and perpetually release more funding to fight the environmental crisis. We believe this new structure delivers on both and we hope it will inspire a new way of doing business that puts people and planet first."
Patagonia has long been known as a conservationist company and has been outspoken on hot-button issues outside of its stores over the years. Patagonia's corporate activism is a large part of its brand identity.
In 2017, the company sued then-President Donald Trump over his administration's move to dramatically shrink two national monuments in Utah.
"The president stole your land," Patagonia's website said at the time. "This is the largest elimination of protected land in American history."
The company emerged as one of the most vocal corporate opponents of Trump's environmental policies.
Last year, Patagonia CEO Ryan Gellert called for companies to join the brand in pressuring Facebook to fix its platforms and the company donated $1 million to voting rights groups in Georgia."""

documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \
    .setInputCols("document") \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(['sentence']) \
    .setOutputCol('token')


embeddings = BertEmbeddings.pretrained(name="electra_large_uncased", lang='en') \
    .setInputCols(['sentence', 'token']) \
    .setOutputCol('embeddings')

ner_model = NerDLModel.pretrained("onto_electra_large_uncased", "en") \
    .setInputCols(['sentence', 'token', 'embeddings']) \
    .setOutputCol('ner')

ner_converter = NerConverter() \
    .setInputCols(['sentence', 'token', 'ner']) \
    .setOutputCol('ner_chunk')

nlp_pipeline = Pipeline(stages=[
    documentAssembler,
    sentence, 
    tokenizer,
    embeddings,
    ner_model,
    ner_converter
])


empty_df = spark.createDataFrame([['']]).toDF('text')
pipeline_model = nlp_pipeline.fit(empty_df)
df = spark.createDataFrame([[text]]).toDF("text")
result = pipeline_model.transform(df)
result.printSchema()
result_df = result.select(F.explode(F.arrays_zip(result.ner_chunk.result, result.ner_chunk.metadata)).alias("cols")) \
      .select(F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"))

print("Named entity recognition")
result_df.show(200,False)

result_df2 = result.select('text',F.explode(F.arrays_zip(result.ner_chunk.result, result.ner_chunk.metadata)).alias("cols")) \
      .select('text',F.expr("cols['0']").alias("chunk"),
              F.expr("cols['1']['entity']").alias("ner_label"))

resultdf_grouped2 = result_df2.groupby('text').agg(F.collect_set('chunk').alias('chunk_grouped'),F.collect_set('ner_label').alias('ner_labels_grouped')).toPandas()

ner_chunks = []
ner_each_items = []
for index,row in resultdf_grouped2.iterrows():
    ner_chunk = row['chunk_grouped']
    ner_chunks.append(ner_chunk)


for item in ner_chunks:
    for each in item:
        ner_each_items.append(each)



print("Relationship extraction")
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref','openie'],memory='8G', be_quiet=False) as client:
    ann = client.annotate(text)
    modified_text = tokenize.sent_tokenize(text)
    for coref in ann.corefChain:
        antecedent = []
        for mention in coref.mention:
            phrase = []
            for i in range(mention.beginIndex, mention.endIndex):
                phrase.append(ann.sentence[mention.sentenceIndex].token[i].word)
            if antecedent == []:
                antecedent = ' '.join(word for word in phrase)
            else:
                anaphor = ' '.join(word for word in phrase)
                modified_text[mention.sentenceIndex] = modified_text[mention.sentenceIndex].replace(anaphor, antecedent)

    modified_text = ' '.join(modified_text)

    ann2 = client.annotate(modified_text)
    #print(ann)
    for sentence in ann.sentence:
        for triple in sentence.openieTriple:
            subject = str(triple.subject)
            relation = str(triple.relation)
            object = str(triple.object)
            if subject in ner_each_items and object in ner_each_items:
                print("relation_triplet:",triple.subject + " " +triple.relation+ " "+triple.object,'\n')

