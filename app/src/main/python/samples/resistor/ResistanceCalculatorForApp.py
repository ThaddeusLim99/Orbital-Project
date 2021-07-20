#simple implementation of resistance calculation
colourCode = {
    "Black": 0,
    "Brown": 1,
    "Red": 2,
    "Orange": 3,
    "Yellow": 4,
    "Green": 5,
    "Blue": 6,
    "Violet": 7,
    "Grey": 8,
    "White": 9,
    "Gold": -1,
    "Silver": -2
}

toleranceCode = { 
    "Brown": 0.01,
    "Red": 0.02,
    "Orange": 0.0005,
    "Yellow": 0.0002,
    "Green": 0.005,
    "Blue": 0.0025,
    "Violet": 0.001,
    "Grey": 0.0001,
    "Gold": 0.05,
    "Silver": 0.1
}

ppmCode = {
    "Black": 250,
    "Brown": 100,
    "Red": 50,
    "Orange": 15,
    "Yellow": 25,
    "Green": 20,
    "Blue": 10,
    "Violet": 5,
    "Grey": 1
}

def threeBandCalc(first, second, m):
    res = (colourCode[first] * 10 + colourCode[second]) * 10**colourCode[m]
    lb = res * 0.8
    ub = res * 1.2
    return res, lb, ub, 20

def fourBandCalc(first, second, m, tol):
    res = (colourCode[first] * 10 + colourCode[second]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub, toleranceCode[tol]*100

def fiveBandCalc(first, second, third, m, tol):
    res = (colourCode[first] * 100 + colourCode[second] * 10 + colourCode[third]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub, toleranceCode[tol]*100

def sixBandCalc(first, second, third, m, tol, ppm):
    res = (colourCode[first] * 100 + colourCode[second] * 10 + colourCode[third]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub, toleranceCode[tol]*100, ppmCode[ppm]

def main(numOfBands, colours):
    if numOfBands == 3:
        results = threeBandCalc(colours[0], colours[1], colours[2])
    elif numOfBands == 4:
        results = fourBandCalc(colours[0], colours[1], colours[2], colours[3])
    elif numOfBands == 5:
        results = fiveBandCalc(colours[0], colours[1], colours[2], colours[3], colours[4])
    elif numOfBands == 6:
        results = sixBandCalc(colours[0], colours[1], colours[2], colours[3], colours[4], colours[5])

    temp = list(results)

    for i in range(len(results)):
        if temp[i] >= 1000000:
            temp[i] /= 1000000
            temp[i] = str(temp[i]) + " MOhms"
        elif temp[i] >= 1000:
            temp[i] /= 1000
            temp[i] = str(temp[i]) + " kOhms"
        else:
            temp[i] = str(temp[i]) + " Ohms"

    result = tuple(temp)

    if numOfBands <= 5:
        print("Resistance:", result[0], "LB:", result[1], "UB:", result[2])
    elif numOfBands == 6:
        print("Resistance:", result[0], "LB:", result[1], "UB:", result[2], "PPM:", results[3])

    return result