import sys
import random
import configparser
from pathlib import Path
import fcntl

def generate_key(length=32):
    """Generate a random key of given length"""
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()-_=+[]{}|;,.<>?/~'
    return ''.join(random.choice(chars) for _ in range(length))

def get_api_keys_file():
    """Read the API keys file location from config.ini"""
    config = configparser.ConfigParser()
    config.read('config.ini')
    return Path(config.get('Auth', 'api_keys_file', fallback='config/api_keys.txt'))

def add_user():
    """Add a new user to the API keys file"""
    api_keys_file = get_api_keys_file()
    
    user_name = input('Enter the username: ')
    key = generate_key()
    
    print(f'\nProposed API key for {user_name}:')
    print(f'{key}\n')
    
    confirm = input('Do you want to add this user? (y/n): ').lower()
    if confirm != 'y':
        print('User addition cancelled.')
        return
    
    api_keys_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(api_keys_file, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
        try:
            f.seek(0)
            existing_users = [line.split(':')[0] for line in f.readlines()]
            
            if user_name in existing_users:
                print(f'Error: User {user_name} already exists.')
                return
            
            f.write(f'{key}:{user_name}\n')
            print(f'User {user_name} added to the API keys file.')
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
    
    print(f'\nAPI Key for {user_name}:')
    print(f'{key}')
    print('\nPlease store this key securely. It will not be displayed again.')

def main():
    add_user()

if __name__ == '__main__':
    main()
