#simple implementation of resistance calculation
colourCode = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
    "grey": 8,
    "white": 9,
    "gold": -1,
    "silver": -2
}

toleranceCode = { 
    "brown": 0.01,
    "red": 0.02,
    "orange": 0.0005,
    "yellow": 0.0002,
    "green": 0.005,
    "blue": 0.0025,
    "violet": 0.001,
    "grey": 0.0001,
    "gold": 0.05,
    "silver": 0.1
}

ppmCode = {
    "black": 250,
    "brown": 100,
    "red": 50,
    "orange": 15,
    "yellow": 25,
    "green": 20,
    "blue": 10,
    "violet": 5,
    "grey": 1
}

def threeBandCalc(first, second, m):
    res = (colourCode[first] * 10 + colourCode[second]) * 10**colourCode[m]
    lb = res * 0.8
    ub = res * 1.2
    return res, lb, ub

def fourBandCalc(first, second, m, tol):
    res = (colourCode[first] * 10 + colourCode[second]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub

def fiveBandCalc(first, second, third, m, tol):
    res = (colourCode[first] * 100 + colourCode[second] * 10 + colourCode[third]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub

def sixBandCalc(first, second, third, m, tol, ppm):
    res = (colourCode[first] * 100 + colourCode[second] * 10 + colourCode[third]) * 10**colourCode[m]
    lb = res * (1 - toleranceCode[tol])
    ub = res * (1 + toleranceCode[tol])
    return res, lb, ub, ppmCode[ppm]

