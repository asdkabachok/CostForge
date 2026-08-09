# CostForge lit test configuration
#
# Run with:
#   lit tests/lit/ --param costforge=build/CostForge.so
#   lit tests/lit/ -v  (verbose)

import lit.formats

config.name = "CostForge"
config.test_format = lit.formats.ShTest()
config.suffixes = ['.ll']

# Substitutions
config.substitutions.append(('%opt', 'opt'))
config.substitutions.append(('%costforge', 
    '-load-pass-plugin=' + config.params.get('costforge', 'build/CostForge.so')))
config.substitutions.append(('%FileCheck', 'FileCheck'))
