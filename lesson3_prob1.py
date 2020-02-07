class Employee:
    #emp_count = 0
    #avg = 0

    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        #Employee.emp_count += 1
        #Employee.avg += salary

    #def sal_avg(self):
     #   print(Employee.avg / Employee.emp_count)


emp = []
print("please enter the number of employees:")
n = int(input())
e = ""
for i in range(n):
    print("please enter the first name of " + str(i + 1) + "th employee")
    name = input()
    print("please enter the family name of " + str(i + 1) + "th employee")
    family = input()
    print("please enter the salary of " + str(i + 1) + "th employee ")
    salary = int(input())
    print("please enter the department of " + str(i + 1) + "th employee ")
    department = input()
    e = Employee(name, family, salary, department)
    emp.append(e)

print("Thank you for entering the details!\n\n")


# for j in range(n):
#     print("The first name of " + str(j + 1) + "th employee")
#     print(emp[j].name)
#     print("The family name of " + str(j + 1) + "th employee")
#     print(emp[j].family)
#     print("The salary of " + str(j + 1) + "th employee")
#     print(emp[j].salary)
#     print("The department of " + str(j + 1) + "th employee")
#     print(emp[j].department)


class Fulltime(Employee):
    pass


full1 = Fulltime('shubh', 'mukherjee', 4000, 'cns')

print(full1.family)

# emp = []
# print("please enter the number of employees:")
# n = int(input())
#
# for i in range(n):
#     print("please enter the first name of " + str(i + 1) + "th employee")
#     name = input()
#     print("please enter the family name of " + str(i + 1) + "th employee")
#     family = input()
#     print("please enter the salary of " + str(i + 1) + "th employee ")
#     salary = int(input())
#     print("please enter the department of " + str(i + 1) + "th employee ")
#     department = input()
#     emp.append(Fulltime(name, family, salary, department))
#
# print("Thank you for entering the details!\n\n")
#
# for j in range(n):
#     print("The first name of " + str(j + 1) + "th employee")
#     print(emp[j].name)
#     print("The family name of " + str(j + 1) + "th employee")
#     print(emp[j].family)
#     print("The salary of " + str(j + 1) + "th employee")
#     print(emp[j].salary)
#     print("The department of " + str(j + 1) + "th employee")
#     print(emp[j].department)


#e.sal_avg()
