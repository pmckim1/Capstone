import os
import time
import uuid

import igraph


def test_line(line):
    filename = os.path.join("wd", str(uuid.uuid4()) + ".net")
    while True:
        with open(filename, "w", encoding="utf-8") as intermediate_file:
            intermediate_file.write("""*Vertices 67883
1 "0 - Continuity IRA admits Brexit day lorry bomb plot"
2 "1 - Brother of Manchester Arena bomber says he was 'shocked' by attack"
3 "2 - Twitter users mock 'ladies fillet' steak on Liverpool menu"
4 "3 - Begum verdict emerges from thin arguments of security v humanity"
{}
*Edges 4
1 2 2.7240674210431015
1 3 2.7240674210431015
1 4 2.7240674210431015
2 3 1.4800995226111608
2 4 1.8692148638617678
3 4 1.6221342969115085
""".format(line))
        try:
            igraph.Graph.Read(filename, 'pajek')
            return True
        except RuntimeError as re:
            print(re)
            print("Too many FDs, lets sleep on it.")
            time.sleep(2)
        except Exception as e:
            print(e)
            return False
        finally:
            os.unlink(filename)


if __name__ == "__main__":
    os.rename("Output/production_output_final.net", "production_output_final.net.unamended")
    with open("Output/production_output_final_unamended.net", 'r', encoding="utf-8") as inf, \
            open("Output/production_output_final.net", 'w', encoding="utf-8") as outf:
        in_vertices = True
        for line in inf:
            amended_line = line
            if len(line) > 1:
                if line[0] == '*' and line[1:].startswith("Vertices"):
                    in_vertices = True
                elif line[0] == '*' and line[1:].startswith("Edges"):
                    in_vertices = False
                elif in_vertices:
                    """
                    passed = test_line(line)
                    if passed:
                        outf.write(amended_line)
                        # print("Accepted line:", amended_line)
                    else:
                        print("Skipping line:", amended_line)
                    """
                    first_term, second_term = line.split(' ', maxsplit=1)
                    # print("Term: ", first_term, second_term)
                    second_term: str = second_term.strip()
                    second_term = second_term.strip("\"")
                    second_term = second_term.replace("\"", "")
                    amended_line = first_term + " \"" + second_term + "\"\n"
                    if first_term == "1233":
                        print(amended_line)
            outf.write(amended_line)
