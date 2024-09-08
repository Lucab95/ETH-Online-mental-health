# Nillion installation check --> https://docs.nillion.com/nillion-sdk-and-tools
curl https://nilup.nilogy.xyz/install.sh | bash
# open new terminal
nilup -V
// Your output should be similar to the below
nilup 22c84830fff3c86beec27a8cb6353d45e7bfb8a4

nilup install latest
nilup use latest
nilup init


# ETH-Online-mental-health
### Inference 

1. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate 
```

2. Install all the required library by:
```
 pip install -r requirements.txt
 ```
3. Create an .env file according to `https://docs.nillion.com/network-configuration` and place it inside `/nillion/quickstart/nada_quickstart_programs`

4. Do nillion-devnet which will print: 🌄 environment file written to /Users/User/.config/nillion/nillion-devnet.env on the last line, open that file and create and paste its components on an .env file on the quickstart_complete/client_code folder
