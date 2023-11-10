import subprocess
import argparse
import json


def construct_parser():
    parser = argparse.ArgumentParser(
        description='Parser for the Sparsey Python wrapper.'
    )

    parser.add_argument(
        '--com_filepath', required=True, type=str,
        help='Path to the CMD / COM file used to configure the Sparsey run.'
    )

    parser.add_argument(
        '--sparsey_cmd_jar', required=False,
        default=''.join(
            [
                'F:\College\Penn\Fall 23\SWENG480\Sparsey\Java app',
                '\Sparsey\SparseyCmdLine\dist\SparseyCmdLine.jar'
            ]
        ), type=str,
        help='Path to the SparseyCMD JAR.'
    )

    return parser


def get_output_folder(com_filepath):
    com_json = json.load(open(com_filepath, 'r'))

    print(com_json)

def execute_sparsey_run(com_filepath, sparsey_cmd_jar):
    output = subprocess.run(
        [
            'java',
            '-jar',
            sparsey_cmd_jar,
            com_filepath
        ], shell=True,
        capture_output=True
    )

    print(output.stdout.decode('utf-8'))
    print(output.stderr.decode('utf-8'))


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()

    output_folder = get_output_folder(args.com_filepath)

    # execute_sparsey_run(args.com_filepath, args.sparsey_cmd_jar)