#pragma once

// This is an intrusive_ptr, which does same thing as shared_ptr but has better cache hit of counter
// Is it right?
#include <atomic>
#include <utility>
namespace oven {

class RefCountable {
private:
    mutable std::atomic<uint32_t> count_{1};
    // we have to count for const objects, too.
public:
    void inc_ref() const noexcept {
        // Here, since increasing reference means that the new intrusive_ptr is referencing the data,
        // We do not need to think about its ordering. It just have to increase.
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    bool dec_ref() const noexcept {
        // Here, decreasing the reference count is a bit different.
        // Suppose we have three threads here.
        // THREAD 1: Write to buffer -> dec_ref() : count 3->2
        // THREAD 2: dec_ref()                    : count 2->1
        // THREAD 3: dec_ref()                    : count 1->0      call delete!
        // Here, due to compiler or CPU reordering or anything,
        // the buffer-write work of THREAD 1 may not have finished until THREAD 3's delete,
        // even though decreas itself is atomic. The atomic does not gurantee the order of the operation.
        // This is why we need memory_release ordering here. the buffer-write work should be done before the delete operation.
        // And, the delete operation cannot be done before the acquire fence.
        // You'll know why.
        if (count_.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            return true;
        }
        return false;
    }

protected:
    ~RefCountable() = default;
};

// intrusive pointer class
// T must inherit the RefCount class
template <typename T>
class intrusive_ptr {
private:
    T* ptr_;

    explicit intrusive_ptr(T* p) noexcept : ptr_(p) {}
    void release() noexcept {
        if (ptr_ && ptr_->dec_ref()) delete ptr_;
    }

    template <typename U, typename... Args>
    friend intrusive_ptr<U> make_intrusive(Args&&...);

public:
    intrusive_ptr() noexcept = default;
    intrusive_ptr(const intrusive_ptr& other) noexcept : ptr_(other.ptr_) {
        if (ptr_) ptr_->inc_ref();
    }
    intrusive_ptr(intrusive_ptr&& other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    intrusive_ptr& operator=(intrusive_ptr other) noexcept { // copy-and-swap
        // What happens here?
        
        // We currently have A now.
        // Suppose that we called operator= with original lavalue ptr named B.
        // Then, the other(lvalue) will be called, and the REFCOUNT of B INCREASES.
        // After that, only the swap happens, WITH NO REFCOUNT INCREASE.
        // After the return, the other is destructed and REFCOUNT of A DECREASES.
        // 
        // Final result is A: refcount -> refcount + 1
        // B: refcount -> refcount - 1

        // Now suppose that we called operator= with rvalue.
        // Then, the move constructor is called and NO REFCOUNT INCREASE happens.
        // Then swap. The other is destructed, and REFCOUNT of A DECREASES.
        // Final result is B: refcount -> refcount. No difference, since it is a move operator=.
        // A: refcount -> refcount - 1. Correct, since the pointer is now not referencing A.
        std::swap(ptr_, other.ptr_);
        return *this;
    }

    ~intrusive_ptr(){ release(); }

    T* operator->() const noexcept {return ptr_;}
    T& operator*() const noexcept {return *ptr_;}
    explicit operator bool() const noexcept {return ptr_;}
    T* get() const noexcept {return ptr_;}

    bool operator==(std::nullptr_t) const noexcept {return ptr_ == nullptr;}
    bool operator!=(std::nullptr_t) const noexcept {return ptr_ != nullptr;}
};

template <typename U, typename... Args>
intrusive_ptr<U> make_intrusive(Args&&... args) {
    return intrusive_ptr<U>(new U(std::forward<Args>(args)...));
}

}// namespace oven