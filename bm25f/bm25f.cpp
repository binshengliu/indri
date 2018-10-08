

/*==========================================================================
 * Copyright (c) 2018 RMIT University.  All Rights Reserved.
 *
 * Use of the Lemur Toolkit for Language Modeling and Information Retrieval
 * is subject to the terms of the software license set forth in the LICENSE
 * file included with this software, and also available at
 * http://www.lemurproject.org/license.html
 *
 *==========================================================================
*/

//
// BM25F
//
// 08 October 2018
//

#include "indri/Parameters.hpp"
#include "QueryBM25F.hpp"

static std::map<std::string, double> parse_field_spec(const std::string& spec) {
  std::map<std::string, double> m;

  int nextComma = 0;
  int nextColon = 0;
  int  location = 0;

  for( location = 0; location < spec.length(); ) {
    nextComma = spec.find( ',', location );
    nextColon = spec.find( ':', location );

    std::string key = spec.substr( location, nextColon-location );
    double value = std::stod(spec.substr( nextColon+1, nextComma-nextColon-1 ));

    m[key] = value;

    if( nextComma > 0 )
      location = nextComma+1;
    else
      location = spec.size();
  }

  return m;
}

static void usage(indri::api::Parameters param) {
  if (!param.exists("index") || !param.exists("query")
      || !param.exists("fieldB") || !param.exists("fieldWt") || !param.exists("k1")) {
    std::cerr << "bm25f usage: " << std::endl
              << "  bm25f -index=myindex -qno=1 -query=myquery -count=1000 -k1=10 -fieldB=title:8,body:2 -fieldWt=title:6,body:1" << std::endl
              << std::endl;
    exit(-1);
  }
}

int main( int argc, char** argv ) {
  try {
    cerr << "Built with " << INDRI_DISTRIBUTION << endl;

    indri::api::Parameters& param = indri::api::Parameters::instance();
    param.loadCommandLine( argc, argv );
    usage( param );

    std::string index = param["index"];
    std::string query = param["query"];
    std::string qno = param.get("qno", "1");
    int requested = param.get("count", 1000);

    int k1 = param.get("k1");
    std::map<std::string, double> fieldB = parse_field_spec(param["fieldB"]);
    std::map<std::string, double> fieldWt = parse_field_spec(param["fieldWt"]);

    // Use docPair vector to record all the fields
    std::vector<std::string> fields;
    for (auto f: fieldB) {
      fields.push_back(f.first);
    }

    QueryBM25F bm25f(index, fields, fieldB, fieldWt, k1, requested);
    bm25f.query(qno, query);
  }
  catch( lemur::api::Exception& e ) {
    LEMUR_ABORT(e);
  } catch( ... ) {
    std::cout << "Caught unhandled exception" << std::endl;
    return -1;
  }

  return 0;
}

