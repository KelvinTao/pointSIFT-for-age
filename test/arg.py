import argparse

def main():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-n','--name', default=' Li ')
    parser.add_argument('-y','--year', default='20')
    args = parser.parse_args()
    print(args)
    name = args.name
    year = args.year
    print('Hello {}  {}'.format(name,year))

if __name__ == '__main__':
    main()
