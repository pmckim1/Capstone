#!/usr/bin/env python2
#Gets the text and some meta data about individual articles
#JB 0208 / TN 1108

#Further ideas:
#
#Articles linked to from the main story page are likely to be related
#Many papers have specifically related articles
#can also get keywords from some pages

#####
#PARAMS, IMPORTS, GLOBALS
#####

#import warc_utils2 as warc_utils

from __future__ import print_function
# Note this is Hanzo warctools (c. 4.9.0) which can be had with
# pip install warctools
# not any of the similarly-named libraries floating around.
# Not functioning under Python 3, sadly
from hanzo.warctools import WarcRecord
from hanzo.httptools import RequestMessage, ResponseMessage
# BeautifulSoup4, to unbugger the incoming text encodings
from bs4 import UnicodeDammit
import re
import sys
import csv
import traceback
import codecs

#list of warc files
infiles = open("infile-list.txt", "r")
directory = "WARCs/"
outdirectory = "ArticleTexts/"
#testing
#print "Testing mode on - just trying one file"
#infiles = ["WEB-20130602165219848-00087-13194~www1.oii.ox.ac.uk~8443.warc"]

#old sites of interest
#sites_of_interest = ("http://www.bbc.co.uk/news/", "http://www.telegraph.co.uk/",
#                     "http://www.express.co.uk/", "http://www.guardian.co.uk/",
#                     "http://www.thesun.co.uk/sol/homepage/", "http://www.dailymail.co.uk/home/index.html",
#                     "http://www.mirror.co.uk/")

sites_of_interest = ("http://www.bbc.co.uk/news/", #"http://www.thesun.co.uk/sol/homepage/",
                     "http://www.express.co.uk/", "http://www.guardian.co.uk/",
                     "http://www.dailymail.co.uk/home/index.html",
                     "http://www.mirror.co.uk/")



#####
#CORE FUNCTIONS: INDEX PAGES
#####

#check to see if something is an article in the express
#couldn't do it all in a lambda? could switch to a regex
def is_express(href):

    chunks = href.split("/")

    for chunk in chunks:

        if chunk.isdigit():
            return True
    return False
    

#determine if a given link is an article link
#based on a predetermined schema for each website
def is_article(link, website):

    #dict of article test functions
    article_tests = {

        #BBC: last character of every article link is a number
        "http://www.bbc.co.uk/news/" : lambda href: href[-1].isdigit(),

        #Telegraph: ends with html
        "http://www.telegraph.co.uk/" : lambda href: ".html" in href,

        #Express: part of the path should be a string of 6 numbers
        "http://www.express.co.uk/" : is_express,

        #Guardian: part of the path should be the year (2013) 
        "http://www.guardian.co.uk/" : lambda href: "2013" in href,

        #Sun: ends with html
        #NB need to get rid of twitter and facebook share buttons
        "http://www.thesun.co.uk/sol/homepage/" : lambda href: ".html" in href,

        #Daily Mail: has the string 'article-' in it
        "http://www.dailymail.co.uk/home/index.html" : lambda href: "article-" in href,

        #Mirror: last seven characters of every article link is a number
        "http://www.mirror.co.uk/" : lambda href: href[-7:].isdigit(),
        
        }

    return article_tests[website](link)


#turn relative (or absolute) links into absolute ones
def abs_link(link, website):

    #dict of relative link replace schemas
    link_reps = {

        #BBC
        "http://www.bbc.co.uk/news/" : "http://www.bbc.co.uk",

        #Telegraph
        "http://www.telegraph.co.uk/" : "http://www.telegraph.co.uk",

        #Express
        "http://www.express.co.uk/" : "http://www.express.co.uk",

        #Guardian
        "http://www.guardian.co.uk/" : "http://www.guardian.co.uk",
        
        #Sun
        "http://www.thesun.co.uk/sol/homepage/" : "http://www.thesun.co.uk",

        #Daily Mail
        "http://www.dailymail.co.uk/home/index.html" : "http://www.dailymail.co.uk",

        #Mirror
        "http://www.mirror.co.uk/" : "http://www.mirror.co.uk"

        }

    if "#" in link:
        link = link.split("#")
        link = link[0]

    if "?" in link:
        link = link.split("?")
        link = link[0]

    #may already be an absolute link
    if not link_reps[website] in link:
        link = link_reps[website] + link

    return link

#Extract a list of article links from a given piece of html
def links(html, website, date):

    global link_prog
    links = {}

    #regex for the links
    for link in re.findall(link_prog, html):

        href = abs_link(link, website)

        #some stuff disappears in this case
        if not href:
            continue

        #check to see if this is actually an article
        if is_article(href, website):

            #might not be the first link of its kind on the page
            if not href in links:
                links[href] = 1
                

    return links


def meta_info(html, site, date):

    #this is the master list
    global link_dictionary

    #get all the links for this specific date
    current_links = links(html, site, date)

    #add / update these links in the dictionary
    for link in current_links:

        if not link in link_dictionary:

            link_dictionary[link] = {}
            link_dictionary[link]["Start"] = date
            link_dictionary[link]["Extracted"] = False
            link_dictionary[link]["Site"] = site
            link_dictionary[link]["Title"] = ""
            
        #update the last recorded time
        link_dictionary[link]["End"] = date

#####
#CORE FUNCTIONS: ARTICLE PAGES
#####

def extract_http_response(record):
    """Parses the content body from an HTTP 'response' record, decoding the
       body to UTF-8 if possible."

    Adapted from github's internetarchive/warctools hanzo/warcfilter.py,
    commit 1850f328e31e505569126b4739cec62ffa444223. MIT licenced."""
    message = ResponseMessage(RequestMessage())
    remainder = message.feed(record.content[1])
    message.close()
    if remainder or not message.complete():
        if remainder:
            print('trailing data in http response for', record.url)
        if not message.complete():
            print('truncated http response for', record.url)

    # Drearily try to extract the encoding from the response MIME type header
    encoding = None
    ctheaders = [x[1] for x in message.header.headers if x[0] == 'Content-Type']
    if len(ctheaders):
        contenttype = ctheaders.pop()
        match = re.search(r'.*;.*charset=([0-9a-zA-Z\-]+).*', contenttype)
        if match:
            encoding = match.group(1)
#       if not encoding:
#           encoding = 'iso-8859-1' # HTTP 1.1 default

    # Guess the correct decoding of the body and then return as a utf-8 coded
    # string.
    # Python2's unicode handling really does suck (the csv module in particular).
    # Unfortunately, the warctools aren't Python3 ready...
    return UnicodeDammit(message.get_body(), [encoding]).unicode_markup.encode('utf-8')


#extract all the paras from an html page
def text_extract(link, sitename, html):

    global title_prog
    global para_prog
    global mark_prog
    global anchor_prog
    global specific_progs

    

    sitename = shortname(sitename)

    spec_prog = ""
    if sitename in specific_progs:
        spec_prog = specific_progs[sitename]
        

##    #could improve with keywords etc.
##    try:

    #get a title, if there is one
    title = re.findall(title_prog, html)
    if len(title) < 1:
        title = ""
    else:
        title = title[0]
    

    story = ""

    #get all paragraphs on the page
    for para in re.findall(para_prog, html):

        #remove any links found within the para, replace with space
        para = re.sub(anchor_prog, " ", para)

        #remove any remaining markup from para, replace with space
        para = re.sub(mark_prog, ' ', para)

        story = story + para

    #get any extra / special schemas
    if spec_prog != "":

        for para in re.findall(spec_prog, html):

            #remove any links found within the para, replace with space
            para = re.sub(anchor_prog, " ", para)

            #remove any remaining markup from para, replace with space
            para = re.sub(mark_prog, ' ', para)

            story = story + para

    #debug
    #print "Debugging on - all html is being written out to err files"
    #error(html, sitename, link)

    return story, title

##    except:
##
##        error(html, sitename, link)
##        
##        return False, False

    
#turn one key value pair in the global link dictionary into a strip for output
#def metainfo(link):
#
#    headersep = "\n####\n"
#    #remove semicolons here
#    link_dictionary[link]["Title"] = link_dictionary[link]["Title"].replace(";", "")
#    vals = ";".join((link, link_dictionary[link]["Site"], link_dictionary[link]["Title"], link_dictionary[link]["Start"], link_dictionary[link]["End"]))
#
#    return headersep + sanitize(vals) + headersep

#turn one key value pair in the global link dictionary into a list of values
def metainfol(link):
    return [link, link_dictionary[link]["Site"], link_dictionary[link]["Title"], link_dictionary[link]["Start"]]#, link_dictionary[link]["End"])

#initialises all the various required files and sets up csv.writer instances
def writer_init(names):

    writer_dict = {}

    for name in names:

        writer_dict[name] = {}

        writer_dict[name] = {}
        writer_dict[name]["out"] = csv.writer(open(outdirectory + name + "-out.csv", "wb"))
        writer_dict[name]["err"] = csv.writer(open(outdirectory + name + "-err.csv", "wb"))

    return writer_dict
        

def error(html, sitename, link):

    e = sys.exc_info()

    global writer_dict
    writer_dict[sitename]["err"].writerow(metainfol(link) + [e[0], html])
#    file_dict[sitename]["err"].write(metainfo(link))
#    file_dict[sitename]["err"].write(str(e[0]) + "\n\n")
#    file_dict[sitename]["err"].write(html)

def file_close():

    global writer_dict
    for name in writer_dict:
        writer_dict[name]["out"].close()
        writer_dict[name]["err"].close()

def writer(html, longname, link):

    global writer_dict

    sitename = shortname(longname)

    writer_dict[sitename]["out"].writerow(metainfol(link) + [html])
#    file_dict[sitename]["out"].write(metainfo(link))
#    file_dict[sitename]["out"].write(sanitize(html))

#long to short name transfer
def shortname(sitename):

    shorts = {

        #BBC
        "http://www.bbc.co.uk/news/" : "bbc",

        #Telegraph
        "http://www.telegraph.co.uk/" : "telegraph",

        #Express
        "http://www.express.co.uk/" : "express",

        #Guardian
        "http://www.guardian.co.uk/" : "guardian",
        
        #Sun
        "http://www.thesun.co.uk/sol/homepage/" : "sun",

        #Daily Mail
        "http://www.dailymail.co.uk/home/index.html" : "mail",

        #Mirror
        "http://www.mirror.co.uk/" : "mirror"

        }

    return shorts[sitename]


#####
#MAIN
#####

#a record of all article links extracted from front pages
link_dictionary = {}

#initialise outfiles
shortnames = ("bbc", "mirror", "guardian", "express", "mail",)#"telegraph", "sun", "mail")
#file_dict = file_init(shortnames)
writer_dict = writer_init(shortnames)

#regexs
para_prog = re.compile('<p>(.*?)</p>', re.DOTALL|re.MULTILINE)
title_prog = re.compile('<title>(.*?)</title>', re.DOTALL|re.MULTILINE)
link_prog = re.compile('href=\"(.*?)\"', re.DOTALL|re.MULTILINE)
mark_prog = re.compile('<[^>]+>', re.DOTALL|re.MULTILINE)
anchor_prog = re.compile('<a(.*?)</a>', re.DOTALL|re.MULTILINE)

specific_progs = {
    "express": re.compile('<p class="storycopy">(.*?)</p>', re.DOTALL|re.MULTILINE),
    "mail": re.compile('<p style=(.*?)</p>', re.DOTALL|re.MULTILINE),
    "bbc": re.compile('<p class="story-body__introduction">(.*?)</p>', re.DOTALL|re.MULTILINE)
    }



#Iterate through the list of WARC files
for infile in infiles:

    filename = directory + infile.strip()

    print("Opening", filename)

#    handle = open(filename, "rb")

    #Iterate through each WARC file extracting individual pages


    handle = WarcRecord.open_archive(filename, mode='rb')
    for record in handle:
        try:
            if record.type == WarcRecord.RESPONSE:

                html = extract_http_response(record)

                if record.url in sites_of_interest:
                    # XXX ???
                    meta_info(html, record.url, record.date)


                elif record.url in link_dictionary:
                    #If we have already got the text, can continue

                    if link_dictionary[record.url]["Extracted"]:
                        continue

                    #else extract the text
                    text, title = text_extract(record.url, link_dictionary[record.url]["Site"], html)
                    
                    #not all extraction paradigms implemented yet
                    if not text:
                        continue

                    link_dictionary[record.url]["Title"] = title

                    #signal that text has been extracted
                    link_dictionary[record.url]["Extracted"] = True

                    #dump the text to file
                    # FIXME XXX
                    writer(text, link_dictionary[record.url]["Site"], record.url)
                    
        except Exception as e:
            print(("Warning: Extraction failed on "+
                   record.url+": "+str(e.message)))
            traceback.print_exc()


  

    handle.close()
    #testing
    #break



#file_close()
#infiles.close()

            

            
