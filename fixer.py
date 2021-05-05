import csv
import sys
import datetime
csv.field_size_limit(sys.maxsize)

def create_dto(cell_contents):
# for index, row in dc_data.iterrows():
    if type(cell_contents) is not str:
        return "Unknown"
    else:
        # Try the various known time formats.
        dtFormat = [
            '%m/%d/%y',
            "%Y-%m-%dT%H:%M:%SZ",
        ]
        # Drop decimal timestamp precision, if it exists.
        cell_contents = cell_contents.split('.')[0]
        for i in dtFormat:
            try:
                dto = datetime.datetime.strptime(cell_contents, i)
                return dto.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                pass
        else:
            print("Failed to parse: {:s}".format(cell_contents))
            return None
            
# with open("ArticleTexts/guardian_clean.csv", 'r') as inf, open("ArticleTexts/guardian_clean-out.csv", 'w') as outf:
with open("./ArticleTexts/medium-boi.csv", 'r') as inf, open("./ArticleTexts/medium-boi-out.csv", 'w') as outf:
        reader = csv.DictReader(inf, delimiter=',')
        writer = csv.writer(outf, delimiter=',')
        for line in reader:
                # Input:
                # t,pub_date,sectionid,sectionname,headline,text
                # Output:
                # url, paperurl, title, date, text
                try:
                        if len(line["text"]) > 131072:
                                print("Skipping line {} (too long)".format(line["t"]))
                                continue
                        time = None
                        try:
                                time = create_dto(line["pub_date"])
                        except Exception as e:
                                print("Skipping line {} (bad date) {}".format(line["t"], line["pub_date"]))
                                print(e)
                                continue
                        # Remove the errant newlines in the headline.
                        headline = line["headline"].replace('\n', ' ')
                        # Remove the errant double quotes that sometimes appear and break the bad encodings that happen later.
                        headline = headline.replace('\"', '')
                        # Remove the non-ascii characters that break encodings later.
                        headline = headline.encode("utf-8").decode("ascii", "ignore")
                        text = line["text"].encode("utf-8").decode("ascii", "ignore")
                        writer.writerow([
                                line["webUrl"],
                                line["sectionid"],
                                headline,
                                time,
                                text,
                        ])
                except csv.Error as e:
                        print(e)
                        print("Skipping.")
