import jsonlines

data_file = 'data/ontoemma.context.test'
output_file = 'data/filtered.test'

keep_data = []
filter_on = 'definition'
# filter_on = 'other_context'

# iterate through data file and filter for pairs of ents with definition or context
with jsonlines.open(data_file) as reader:
    for obj in reader:
        if len(obj['source_ent'][filter_on]) > 0 and len(obj['target_ent'][filter_on]) > 0:
            keep_data.append(obj)

# write filtered data to out file
with jsonlines.open(output_file, mode='w') as writer:
    for line in keep_data:
        writer.write(line)
