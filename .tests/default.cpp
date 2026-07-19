#include <iostream>

int basic_test()
{
    return 0;
}

int main()
{
    try {
        return basic_test();
    } catch (std::exception& ex) {
        std::cerr << "[exception]: " << ex.what() << std::endl;
        return -1;
    }
}
