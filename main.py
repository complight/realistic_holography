import sys
import optics
import argparse
from odak.tools import load_dictionary

__title__ = 'Thin headset'

def prepare(settings_filename='./settings/sample.txt'):
    settings = load_dictionary(settings_filename)
    return settings

def main():
    settings_fn = './settings/sample.txt'
    parser      = argparse.ArgumentParser(description=__title__)
    parser.add_argument(
                        '--settings',
                        type=argparse.FileType('r'),
                        help='Filename for the settings files. Default is {}'.format(settings_fn)
                       )
    args        = parser.parse_args()
    if type(args.settings) != type(None):
        settings_fn = str(args.settings.name)
    settings = prepare(settings_filename=settings_fn)
    optics.start(settings)
    return True

if __name__ == '__main__':
    sys.exit(main())
