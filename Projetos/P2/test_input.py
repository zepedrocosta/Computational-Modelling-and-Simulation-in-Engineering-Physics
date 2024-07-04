from colorama import Fore

print(Fore.MAGENTA + "Space Module Reentry Simulation - SMCEF 23/24")
print("==================================") 
print("Insert the simulation mode:")
print(Fore.BLUE + "1 - Automatic:")
print("This mode will run all the values for v0 between 0 and 15000 m/s and all the values for alpha between 0 and 15 degrees.")
print(Fore.GREEN + "2 - Manual")
print("This mode will ask for the values of v0 and alpha.")
user_input = input(Fore.YELLOW + "Please enter the mode (1/2): " + Fore.RESET)

if(user_input == "1"):
    print(Fore.MAGENTA + "Automatic mode selected.")
    print("Running simulation..." + Fore.RESET)
elif(user_input == "2"):
    print(Fore.MAGENTA + "Manual mode selected.")
    v0 = input(Fore.CYAN + "Please enter the initial velocity (m/s): " + Fore.RESET)
    alpha = input(Fore. GREEN + "Please enter the initial angle (degrees): " + Fore.RESET)
    print(Fore.MAGENTA + "Running simulation..." + Fore.RESET)
else:
    print(Fore.RED + "Invalid input. Exiting..." + Fore.RESET)
    exit(1)

