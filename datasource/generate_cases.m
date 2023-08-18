close all;
clear all;

load("ucibp\Part_1.mat");
load("ucibp\Part_2.mat");
load("ucibp\Part_3.mat");
load("ucibp\Part_4.mat");

cell = {Part_1{1:3000}, Part_2{1:3000}, Part_3{1:3000}, Part_4{1:3000}};

mkdir("cases")

for i = 1:12000
    i
    data = cell{i};
    caseId = i;
    save(sprintf("cases\\case-%d.mat", i), 'data', "caseId");
end