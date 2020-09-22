def not_break(sen):
    return (sen != '\n' and sen != '\u3000' and  sen != '' and not sen.isspace())
def filter_data(ini_data):
    # ini_data是由句子组成的string
    new_data = list(filter(not_break, [data.strip() for data in ini_data]))
    return new_data