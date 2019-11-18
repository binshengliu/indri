

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

#include "indri/TokenizerFactory.hpp"
#include "indri/ParserFactory.hpp"
#include "indri/Parameters.hpp"
#include "indri/IndriParser.hpp"
#include "indri/ParsedDocument.hpp"
#include "indri/StemmerFactory.hpp"
#include "indri/NormalizationTransformation.hpp"
#include "indri/UTF8CaseNormalizationTransformation.hpp"
#include "../external/argparse/include/argparse.hpp"
#include "indri/FileClassEnvironmentFactory.hpp"

int main(int argc, char **argv) {
  try {
    argparse::ArgumentParser program(argv[0]);

    program.add_argument("--type")
        .help("txt (default), html, "
              "xml.\n\t\tExample: <script>code</"
              "script><title>00000-NRT-RealEstate's "
              "Homepage "
              "Startup</title><body>content</"
              "body>\n\t\ttxt: script code script title 00000 nrt realestate "
              "homepage startup title body content body\n\t\thtml: 00000 nrt "
              "realestate homepage startup content\n\t\txml: code 00000 nrt "
              "realestate homepage startup content")
        .default_value(std::string("txt"))
        .action([](const std::string &value) {
          static const std::vector<std::string> choices = {"txt", "html", "xml"};
          if (std::find(choices.begin(), choices.end(), value) !=
              choices.end()) {
            return value;
          }
          return std::string{"txt"};
        });

    program.add_argument("--stemmer")
        .help("none, krovetz (default), porter, arabic")
        .default_value(std::string("krovetz"))
        .action([](const std::string &value) {
          static const std::vector<std::string> choices = {"none", "krovetz",
                                                           "porter", "arabic"};
          if (std::find(choices.begin(), choices.end(), value) !=
              choices.end()) {
            return value;
          }
          return std::string{"krovetz"};
        });

    try {
      program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
      std::cerr << err.what() << std::endl;
      std::cerr << "Built with " << INDRI_DISTRIBUTION << endl;
      std::cerr << program;
      exit(0);
    }

    std::string type = program.get<std::string>("--type");
    indri::parse::FileClassEnvironmentFactory fileClassFactory;
    indri::parse::FileClassEnvironment *fileEnv = fileClassFactory.get(type);

    std::vector<indri::parse::Transformation*> transformations;
    indri::api::Parameters empty;
    transformations.push_back( new indri::parse::NormalizationTransformation() );
    transformations.push_back( new indri::parse::UTF8CaseNormalizationTransformation() );
    std::string stemmer = program.get<std::string>("--stemmer");
    if (stemmer != "none") {
      transformations.push_back(indri::parse::StemmerFactory::get(stemmer, empty));
    }

    std::string line;
    while(std::getline(std::cin, line)){
      indri::parse::UnparsedDocument document = {.text = line.c_str(), .textLength = line.length(), .content=line.c_str(), .contentLength = line.length()};
      indri::parse::TokenizedDocument* tokenized = fileEnv->tokenizer->tokenize(&document);
      indri::api::ParsedDocument *parsed = fileEnv->parser->parse( tokenized );
      for (size_t i = 0; i < transformations.size(); i++) {
        parsed = transformations[i]->transform(parsed);
      }

      // std::cout << parsed->positions.size() << " " << parsed->terms.size() << std::endl;
      // std::cout << parsed->text << std::endl;
      // std::cout << parsed->content << std::endl;
      for (size_t i = 0; i < parsed->terms.size(); ++i) {
        if (i < parsed->terms.size() - 1) {
          std::cout << parsed->terms[i] << " ";
        } else {
          std::cout << parsed->terms[i] << std::endl;
        }
      }
    }
  }
  catch( lemur::api::Exception& e ) {
    LEMUR_ABORT(e);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  } catch( ... ) {
    std::cout << "Caught unhandled exception" << std::endl;
    return -1;
  }

  return 0;
}
