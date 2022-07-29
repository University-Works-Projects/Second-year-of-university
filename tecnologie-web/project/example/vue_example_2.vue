<script>
export default {
  data() {
    return {
      count: 0
    }
  },
  methods: {
    increment() {
      this.count++
    }
  }
}
</script>

<template>
  <button v-on:click="increment">count is: {{ count }}</button>
  <button @click="increment">count is: {{ count }}</button> <!-- Shortcut -->
</template>