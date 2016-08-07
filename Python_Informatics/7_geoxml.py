
# Michael Galarnyk
# Assignment 7

# “Google Geocoding”
# 1. Change either the geojson.py or geoxml.py program to print out the two-character country code from the retrieved data.
# 2. Add error checking so your program does not traceback if the country code is not there.
# 3. Use the program to search for “Pacific Ocean” and make sure that it can handle locations that
# are not in any country.

import urllib
import json

serviceurl = 'http://maps.googleapis.com/maps/api/geocode/json?'

while True:
    address = raw_input('Enter location: ')
    if len(address) < 1 : break

    url = serviceurl + urllib.urlencode({'sensor':'false', 'address': address})
    print 'Retrieving', url
    uh = urllib.urlopen(url)
    data = uh.read()
    print 'Retrieved',len(data),'characters'

    try: js = json.loads(str(data))
    except: js = None
    if 'status' not in js or js['status'] != 'OK':
        print '==== Failure To Retrieve ===='
        print data
        continue

    print json.dumps(js, indent=4)
    '''
    lat = js["results"][0]["geometry"]["location"]["lat"]
    lng = js["results"][0]["geometry"]["location"]["lng"]
    print 'lat',lat,'lng',lng
    ''' #not necessary for this assignment
    location = js['results'][0]['formatted_address']
    print location

    results = js['results'][0]
    address_components = results["address_components"]
    country = 0;
    for each_dict in address_components:
        types = each_dict["types"]
        if types == ["country", "political"]:
            country = 1;
            print "The two character country  code is:", each_dict["short_name"]

    if country == 0:
        print "Location isn't in any country"
