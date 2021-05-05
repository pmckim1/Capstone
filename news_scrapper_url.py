#%%
import csv
import json
import requests

def remove_none(in_dict):
    return {k: v for k, v in in_dict.items() if v is not None}


class GuardianAccess(object):
    def __init__(self):
        #self.api_key = "6f0112ea-0cb0-44bf-895b-445fcf3ec7aa"
        #self.api_key = "65acdd88-bfca-4fa5-85a7-ab4f3ab0c2c4"
        #self.api_key = "80ebd56e-d271-454a-8060-2c30542ce2b7"
        #self.api_key ="4a8e0057-83f3-4c45-b64e-5997f49dc24b"
       #self.api_key = "c3444211-b8f4-49b6-bf07-86c8cfd4855f"
       #self.api_key ="7919cdd7-cd71-48c1-9c65-d7e203cd80bc"
       self.api_key = "c5cdbf4a-443c-443d-8724-ccd54d62aeb0"
       #self.api_key = "6c154206-1a0f-42fe-aa05-79688058af0f"
       #self.api_key = "714dca6e-d1f1-480a-bf67-9a6d3ee12668"
       #self.api_key ="245a4d63-06ef-4cfc-bfec-3f0668a062b4"
       #self.api_key="2f96d936-8c38-4978-b992-434fdc593a6c"
       #self.api_key="b7cff1c5-0483-4599-b6fe-df4e813f6e2f"
       
       
    def search(self, session, section, from_date, to_date, page=1):
        params = {
            "api-key": self.api_key,
            "query-fields": "body",
            "from-date": from_date,
            "to-date": to_date,
            "section": section,
            "page": page,
        }
        params = remove_none(params)

        response = session.get(
            "https://content.guardianapis.com/search",
            params=params,
        )
        try:
            search_response = json.loads(response.text)
            # Check to see if 'response' exists, or if there is an error.
            response = search_response["response"]
        # print(response.request.url)
        # print(response)
            print(json.dumps(search_response, indent='\t'))
        except Exception as e:
            print("Key error:")
            print(e)
            print("Actual API error:")
            print(response.text)
            print(json.loads(response.text))
            return None
        return search_response

    def download_from_search(self, session, search_result):
        print("Attempting download")
        response = session.get(
            search_result["apiUrl"],
            params={
                "api-key": self.api_key,
                "show-blocks": "body",
            }
        )

        # print(json.dumps(json_response, indent='\t'))

        try:
            json_response = json.loads(response.text)
            bodyTextSummary = json_response["response"]["content"]["blocks"]["body"][0]["bodyTextSummary"]
        except Exception as e:
            print("Key error:")
            print(e)
            print("Actual API error:")
            print(response.text)
            print(json.loads(response.text))
            return None
        return bodyTextSummary

    def search_and_download(self, session):
        for section in ["business"]:
            search_response = self.search(session, section=section, from_date="2015-01-01", to_date="2020-01-23")
            print(json.dumps(search_response, indent='\t'))

            with open('little_set34.csv', 'w', newline='', encoding="utf-8") as csvfile:
                fieldnames = ['pub_date', 'sectionId', 'sectionName', 'webTitle', 'text', 'webUrl', 'apiUrl'] 
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
                writer.writeheader()
                
                if search_response:
                    pages = search_response["response"]["pages"]
                else:
                    pages = []
                for page in range(pages):
                    page = page + 1
                    search_response = self.search(session, section=section, from_date="2015-01-01", to_date="2020-01-23", page=page)
                    # print(json.dumps(search_response, indent='\t'))
                    if search_response:
                        results = search_response["response"]["results"]
                    else:
                        results = []
                    for result in results:
                        webTitle = result["webTitle"]
                        #if "blog".strip("\"") in webTitle:
                            #print("Skipping, because blog was in title: {}".format(webTitle))
                            #continue
                        skip_terms_hl = ["blog", "obituary", "letter"]
                        if any(term.lower() in result["webTitle"].lower() for term in skip_terms_hl):
                            print("Skipping headline {} becasue it contained a forbidden term.".format(result["webTitle"]))
                            continue
                        '''
                        skip_terms_hl = ["world"]
                        if any(term.lower() not in result["sectionId"].lower() for term in skip_terms_hl):
                            print("Skipping headline {} becasue it missed a required term.".format(result["sectionId"]))
                            continue
                        '''
                        bodyTextSummary = self.download_from_search(session, result)
                        webTitle = result["webTitle"]
                        print("Saving: {}".format(webTitle))
                        writer.writerow(
                            {
                                "pub_date": result["webPublicationDate"],
                                "webTitle": result["webTitle"],
                                "sectionId": result["sectionId"],
                                "sectionName": result["sectionName"],
                               # "production_office": result["production-office"],
                                "text": bodyTextSummary,
                                "webUrl" : result["webUrl"],
                                "apiUrl" : result["apiUrl"]
                            }
                        )
                    


if __name__ == "__main__":
    ga = GuardianAccess()

    with requests.Session() as session:
        ga.search_and_download(session)

print("complete")

# %%
'''
# test encding:
if __name__ == "__main__":
        params = {
            "api-key": "80ebd56e-d271-454a-8060-2c30542ce2b7",
            "query-fields": "body",
            "from-date": "2019-01-01",
            "to-date": "2020-01-01",
            "section": "world",
            "page": 1,
        }
        params = remove_none(params)

        response = session.get(
            "https://content.guardianapis.com/search",
            params=params,
        )
        print("Encoding", response.encoding)
        try:
            search_response = json.loads(response.text)
            # Check to see if 'response' exists, or if there is an error.
            response = search_response["response"]
            print(response)
        # print(response.request.url)
        # print(response)
        # print(json.dumps(json_response, indent='\t'))
        except Exception as e:
            print("Key error:")
            print(e)
            print("Actual API error:")
            print(response.text)
            print(json.loads(response.text))
            
        print("")
            
        search_result = search_response["response"]["results"][0]
        response = session.get(
            search_result["apiUrl"],
            params={
                "api-key": "80ebd56e-d271-454a-8060-2c30542ce2b7",
                "show-blocks": "body",
            }
        )
        print(json.dumps(response.json(), indent='\t'))
        print("Encoding", response.encoding)

'''
# %%
